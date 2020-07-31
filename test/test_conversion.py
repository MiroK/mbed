from mbed.conversion import mesh_from_gmshModel
import dolfin as df
import numpy as np
import pytest
import gmsh


def test_3d():
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
    gmsh.initialize([])
    
    gmsh.option.setNumber("General.Terminal", 1)

    # Add components to model
    for point in points: model.geo.addPoint(*point)

    lines = [(1, 2), (2, 3), (3, 4), (4, 1),
             (5, 6), (6, 7), (7, 8), (8, 5),
             (1, 5), (2, 6), (3, 7), (4, 8)]

    lines_ = []
    for lidx, line in enumerate(lines, 1):
        lines_.append(model.geo.addLine(*line, tag=lidx))

    surfs = [(1, 2, 3, 4), (5, 6, 7, 8),
             (1, 10, -5, -9), (2, 11, -6, -10),
             (11, 7, -12, -3), (12, 8, -9, -4)]

    plane_tags = []
    for sidx, surf in enumerate(surfs, 1):
        tag = model.geo.addCurveLoop(surf, tag=sidx)
        plane_tags.append(model.geo.addPlaneSurface([tag]))
        
    surface_loop = [model.geo.addSurfaceLoop(plane_tags)]
    volume = model.geo.addVolume(surface_loop)

    model.addPhysicalGroup(1, lines_, 23)

    model.addPhysicalGroup(2, plane_tags[0:3], 2)
    model.addPhysicalGroup(2, plane_tags[3:4], 32)
    model.addPhysicalGroup(2, plane_tags[4:5], 22)
        
    model.addPhysicalGroup(3, [volume], 42)

    model.geo.synchronize()

    # gmsh.fltk.initialize()
    # gmsh.fltk.run()
    
    model.mesh.generate(3)

    # Finally
    mesh, foos = mesh_from_gmshModel(model, include_mesh_functions=-1)

    # We get the cells and nodes right
    elm, nodes = model.mesh.getElementsByType(4)
    assert len(elm) == mesh.num_cells()
    assert len(set(nodes)) == mesh.num_vertices()
    # We have all the mappings
    assert set((1, 2, 3)) == set(foos.keys())

    # Volume
    vol_tags = np.unique(foos[3].array())
    assert len(vol_tags) == 1
    assert vol_tags[0] == 42

    coords = mesh.coordinates()

    # Surfaces
    mesh.init(2, 0)
    f2v = mesh.topology()(2, 0)
    for f in range(mesh.num_entities(2)):
        mid = np.mean(coords[f2v(f)], axis=0)
        if foos[2][f] == 2:
            assert df.near(mid[1], 0) or df.near(mid[2]*(1-mid[2]), 0)
        elif foos[2][f] == 32:
            assert df.near(mid[0], 1)
        elif foos[2][f] == 22:
            assert df.near(mid[1], 1)
        else:
            assert foos[2][f] == 0

    okay_edge = lambda mid: np.array([[df.near(xi, 0) for xi in mid],
                                      [df.near(xi, 1) for xi in mid],
                                      [0 < xi < 1 for xi in mid]])
                                      
    
    # Edge
    mesh.init(1)
    e2v = mesh.topology()(1, 0)
    for edge in range(mesh.num_entities(1)):
        if foos[1][edge] == 23:
            mid = np.mean(coords[e2v(edge)], axis=0)
            assert all([df.near(xi, 0) or df.near(xi, 1) or df.between(xi, (0, 1)) for xi in mid])

    
    gmsh.clear()
    
            
def test_2d():
    x0, y0 = np.zeros(2)
    x1, y1 = np.ones(2)

    points = [(x0, y0, 0),
              (x1, y0, 0),
              (x1, y1, 0),
              (x0, y1, 0)]

    # Add points
    model = gmsh.model
    gmsh.initialize([])

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

    model.addPhysicalGroup(2, plane_tags[0:1], 2)  # y < x
    model.addPhysicalGroup(2, plane_tags[1:2], 3)  # y > x
        
    model.addPhysicalGroup(1, lines[0:2], 22)  # y = 0, x = 1
    model.addPhysicalGroup(1, lines[2:4], 42)  # y = 1, x = 0

    model.addPhysicalGroup(0, [1, 2, 3, 4], 3)  

    model.geo.synchronize()

    model.mesh.generate(2)

    # Finally
    mesh, foos = mesh_from_gmshModel(model, include_mesh_functions=-1)
    
    # We get the cells and nodes right
    elm, nodes = model.mesh.getElementsByType(2)
    assert len(elm) == mesh.num_cells()
    assert len(set(nodes)) == mesh.num_vertices()

    # We have all the mappings
    assert set((0, 1, 2)) == set(foos.keys())

    coords = mesh.coordinates()
    # Cell f
    for cell in df.cells(mesh):
        x, y = cell.midpoint().array()[:2]
        if foos[2][cell] == 2:
            assert y < x
        else:
            assert foos[2][cell] == 3, foos[2][cell]
            assert x < y

    # Facet f
    mesh.init(1)
    f2v = mesh.topology()(1, 0)
    for facet in range(mesh.num_entities(1)):
        x, y = np.mean(coords[f2v(facet)], axis=0)[:2]
        if foos[1][facet] == 22:
            assert df.near(y, 0) or df.near(x, 1)
        
        elif foos[1][facet] == 42:
            assert df.near(y, 1) or df.near(x, 0)
        else:
            assert df.between(x, (0, 1)) and df.between(y, (0, 1))
            assert foos[1][facet] == 0

    # Vertex
    vf = foos[0].array()
    nodes, = np.where(vf == 3)
    assert len(nodes) == 4

    coords = coords[nodes]
    for x in coords:
        assert df.near(x[0], 0) or df.near(x[0], 1)
        assert df.near(x[1], 0) or df.near(x[1], 1)
    
    gmsh.clear()
