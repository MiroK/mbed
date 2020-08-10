# In memory conversion from GMSH to dolfin
import mbed.utils as utils
import dolfin as df
import numpy as np
import itertools
import tqdm
import ufl


code='''
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshEditor.h>
#include <dolfin/mesh/CellType.h>
#include <dolfin/mesh/MeshTopology.h>
#include <dolfin/mesh/MeshConnectivity.h>
#include <dolfin/mesh/MeshValueCollection.h>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <map>

namespace dolfin {
  // Fills a SIMPLICIAL mesh
  void fill_mesh(const Array<double>& coordinates,
                 const Array<std::size_t>& cells, 
                 const int tdim, 
                 const int gdim, 
                 std::shared_ptr<Mesh> mesh)
  {
     int nvertices = coordinates.size()/gdim;     

     int nvertices_per_cell = tdim + 1;
     int ncells = cells.size()/nvertices_per_cell;   

     MeshEditor editor;
     if (tdim == 1){
         editor.open(*mesh, CellType::Type::interval, tdim, gdim);
     }
     else if (tdim == 2){
         editor.open(*mesh, CellType::Type::triangle, tdim, gdim);
     }
     else{
         editor.open(*mesh, CellType::Type::tetrahedron, tdim, gdim);
     }

     editor.init_vertices(nvertices);
     editor.init_cells(ncells);

     std::vector<double> vertex(gdim);
     for(std::size_t index = 0; index < nvertices; index++){
         for(std::size_t i = 0; i < gdim; i++){
             vertex[i] = coordinates[gdim*index  + i];
         }
         editor.add_vertex(index, vertex);
     }

     std::vector<std::size_t> cell(nvertices_per_cell);
     for(std::size_t index = 0; index < ncells; index++){
         for(std::size_t i = 0; i < nvertices_per_cell; i++){
             cell[i] = cells[nvertices_per_cell*index  + i];
         }
         editor.add_cell(index, cell);
     }

     editor.close();
  }
};
'''
module = df.compile_extension_module(code)



def make_mesh(vertices, cells, cell_type=None):
    '''Mesh from data by MeshEditor'''
    # Decide tetrahedron/triangle
    mesh = df.Mesh()
    assert mesh.mpi_comm().size == 1
    
    if cell_type is None:
        if len(cells[0]) == 3:
            gdim = vertices.shape[1]
            module.fill_mesh(vertices.flatten(), cells.flatten(), 2, gdim, mesh)
            return mesh
        
        if len(cells[0]) == 4 and vertices.shape[1] == 3:
            module.fill_mesh(vertices.flatten(), cells.flatten(), 3, 3, mesh)
            return mesh
            
        raise ValueError(
            'cell_type cannot be determined reliably %d %d %d' % (
        (len(cells[0]), ) + vertices.shape))

    if cell_type.cellname() == 'triangle':
        module.fill_mesh(vertices.flatten(), cells.flatten(), 2, cell_type.geometric_dimension, mesh)
        return mesh

    if cell_type.cellname() == 'tetrahedron':
        module.fill_mesh(vertices.flatten(), cells.flatten(), 3, cell_type.geometric_dimension, mesh)
        return mesh

    # Fallback to python
        
    gdim = cell_type.geometric_dimension()
    assert vertices.shape[1] == gdim

    tdim = cell_type.topological_dimension()

    editor = df.MeshEditor()

    editor.open(mesh, str(cell_type), tdim, gdim)            

    editor.init_vertices(len(vertices))
    editor.init_cells(len(cells))

    for vi, x in enumerate(vertices): editor.add_vertex(vi, x)

    for ci, c in enumerate(cells): editor.add_cell(ci, *c)
    
    editor.close()

    return mesh


def mesh_from_gmshModel(model, include_mesh_functions=-1):
    '''
    Return mesh, [mesh functions for tags]
    
    include_mesh_functions = None ... {} 
                             -1  ... all
                             >=0 ... tags for that entity
    '''
    reorder = utils.Timer('Reordering GMSH', 2)
    etypes = set(model.mesh.getElementTypes())
    # Pick highest element; also vertices per cell
    if 4 in etypes:
        etype, tdim, vtx_per_cell = 4, 3, 4
    else:
        assert 2 in etypes, etypes
        etype, tdim, vtx_per_cell = 2, 2, 3

    # All the nodes
    _, vertices, _ = model.mesh.getNodes()
    vertices = np.array(vertices).reshape((-1, 3))  # Gmsh does not do 2d

    if model.getDimension() == 2:
        vertices = vertices[:, 0:2]

    # Get elements encoded in gmsh vertex numbering
    _, cells = model.mesh.getElementsByType(etype)
    # Which vertices are needed to build desired elements
    vtx_idx = np.fromiter(set(cells), dtype='uintp')
    vertices = vertices[vtx_idx-1]  
    # We are now enforcing dolfin numbering and need a map    
    dolfin_map = {gi:di for di, gi in enumerate(vtx_idx)}
    cells = np.fromiter((dolfin_map[gi] for gi in cells), dtype='uintp').reshape((-1, vtx_per_cell))

    reorder.done()

    make = utils.Timer('Building mesh with mesh editor', 2)
    mesh = make_mesh(vertices, cells, cell_type=None)
    make.done()

    q = df.MeshQuality.radius_ratio_min_max(mesh) + (min(c.volume() for c in df.cells(mesh)), )
    utils.print_green('Mesh quality %g, %g, %g' % q)

    # Let's see about tags
    if include_mesh_functions is None:
        include_mesh_functions = []
    elif include_mesh_functions == -1:
        include_mesh_functions = list(range(tdim+1))
    else:
        assert isinstance(include_mesh_functions, int)
        include_mesh_functions = [include_mesh_functions]

    dim_tags = model.getPhysicalGroups()
    node_tags = np.zeros(mesh.num_vertices(), dtype=int)
    
    mesh_fs = {}
    for dim, tags in itertools.groupby(dim_tags, key=utils.first):
        if dim not in include_mesh_functions: continue

        dim > 0 and mesh.init(dim)
        dim > 0 and mesh.init(dim, 0)
            
        mesh_fs[dim] = df.MeshFunction('size_t', mesh, dim, 0)
        array = mesh_fs[dim].array()

        tag_timer = utils.Timer('Tagging %d entities of dim %d' % (mesh.num_entities(dim), dim), 2)
        e2v = mesh.topology()(dim, 0)
        tags = [p[1] for p in tags]
        for tag in tqdm.tqdm(tags):
            assert tag > 0
            
            tagged_entities = tag_entities(model, dim, tag, e2v, dolfin_map, node_tags, array)
                    
        tag_timer.done()
    return mesh, mesh_fs


def tag_entities(model, dim, tag, e2v, node_map, node_tags, array):
    '''Entites of topological dimension `dim` of mesh from the model that have `tag`'''
    entities = model.getEntitiesForPhysicalGroup(dim, tag)

    for entity in entities:
        # Pick nodes on tagged model entities
        tagged_nodes = model.mesh.getNodes(dim, entity, includeBoundary=True)[0]
        # Set in dolfin numbering
        node_tags[[node_map[t] for t in tagged_nodes]] = 1
        # Actual cell tag is determined by its vertices
        if dim > 0:
            # Unseen
            maybe,  = np.where(array == 0)
            # We want those have all 1 as vertices [[1, 1, 1, 1], [1, 1, 0, 1], ... ]            
            tagged_idx,  = np.where(np.prod(node_tags[map(e2v, maybe)], axis=1) == 1)
            array[maybe[tagged_idx]] = tag
        # Vertices decide themselves
        else:
            array[node_tags == 1] = tag
        node_tags.fill(0)
    return array

# --------------------------------------------------------------------

if __name__ == '__main__':
    import gmsh, sys

    # Example of in memory
    if False:
        x0, y0, z0 = np.zeros(3)
        x1, y1, z1 = np.ones(3)

        points = [(x0, y0, z0),
                  (x1, y0, z0),
                  (x1, y1, z0),
                  (x0, y1, z0),
                  (x0, y0, z1),
                  (x1, y0, z1),
                  (x1, y1, z1),
                  (x0, y1, z1)]

        # Add points
        model = gmsh.model
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("General.Terminal", 1)

        # # Add components to model
        for point in points: model.geo.addPoint(*point)

        lines = [(1, 2), (2, 3), (3, 4), (4, 1),
                 (5, 6), (6, 7), (7, 8), (8, 5),
                 (1, 5), (2, 6), (3, 7), (4, 8)]
        
        for lidx, line in enumerate(lines, 1):
            model.geo.addLine(*line, tag=lidx)

        surfs = [(1, 2, 3, 4), (5, 6, 7, 8),
                 (1, 10, -5, -9), (2, 11, -6, -10),
                 (11, 7, -12, -3), (12, 8, -9, -4)]

        plane_tags = []
        for sidx, surf in enumerate(surfs, 1):
            tag = model.geo.addCurveLoop(surf, tag=sidx)
            plane_tags.append(model.geo.addPlaneSurface([tag]))

        surface_loop = [model.geo.addSurfaceLoop(plane_tags)]
        volume = model.geo.addVolume(surface_loop)

        model.addPhysicalGroup(2, plane_tags[0:3], 2)
        model.addPhysicalGroup(2, plane_tags[3:4], 32)
        model.addPhysicalGroup(2, plane_tags[4:5], 22)
        
        model.addPhysicalGroup(3, [volume], 42)

        model.geo.synchronize()
        model.mesh.generate(3)

        # Finally
        mesh, foos = mesh_from_gmshModel(model, include_mesh_functions=-1)
        df.File('xxx.pvd') << mesh

        for d, f in foos.items():
            df.File('fx_%d.pvd' % d) << f

        
    if True:
        x0, y0 = np.zeros(2)
        x1, y1 = np.ones(2)

        points = [(x0, y0, 0),
                  (x1, y0, 0),
                  (x1, y1, 0),
                  (x0, y1, 0)]

        # Add points
        model = gmsh.model
        gmsh.initialize(sys.argv)

        gmsh.option.setNumber("General.Terminal", 1)

        # # Add components to model
        for point in points: model.geo.addPoint(*point)
        #      3<
        # 4v   5/     2^
        #      1>
        lines = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
        
        for lidx, line in enumerate(lines, 1):
            model.geo.addLine(*line, tag=lidx)

        surfs = [(1, 2, -5), (5, 3, 4)]

        plane_tags = []
        for sidx, surf in enumerate(surfs, 1):
            tag = model.geo.addCurveLoop(surf, tag=sidx)
            plane_tags.append(model.geo.addPlaneSurface([tag]))

        model.addPhysicalGroup(2, plane_tags[0:1], 2)
        model.addPhysicalGroup(2, plane_tags[1:2], 3)
        
        model.addPhysicalGroup(1, lines[0:2], 22)
        model.addPhysicalGroup(1, lines[2:4], 42)

        model.addPhysicalGroup(0, [1, 2, 3, 4], 3)

        model.geo.synchronize()
        model.mesh.generate(2)

        #gmsh.fltk.initialize()
        #gmsh.fltk.run()

        # Finally
        mesh, foos = mesh_from_gmshModel(model, include_mesh_functions=-1)
        df.File('xxx.pvd') << mesh

        for d, f in foos.items():
            df.File('fx_%d.pvd' % d) << f
