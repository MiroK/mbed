from mbed.generation import make_line_mesh
import dolfin as df
import numpy as np
import os


def curve_isected_cells(curve_f, predicate=lambda x: x > 0):
    '''Cell intersected by curve.'''
    #  W e count the intersection using vertices of the edges
    assert curve_f.dim() == 1
    mesh = curve_f.mesh()
    
    edges, = np.where(predicate(curve_f.array()))

    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    vertices = np.unique(np.hstack(list(map(e2v, edges))))

    tdim = mesh.topology().dim()
    mesh.init(0, tdim)
    v2c = mesh.topology()(0, tdim)
    
    return np.unique(np.hstack(list(map(v2c, vertices))))


def layer_neighbor_cell_generator(tagged_cells):
    '''Peel next layer of neighbors'''
    # Neighbors are computed using vertex connectivity
    # Immediate neighbors get tag 2 etc ...
    tdim = tagged_cells.dim()
    mesh = tagged_cells.mesh()
    
    assert tdim == mesh.topology().dim()

    mesh.init(0, tdim)
    mesh.init(tdim, 0)
    
    v2c, c2v = mesh.topology()(0, tdim), mesh.topology()(tdim, 0)
    
    values = tagged_cells.array()
    assert set(values) == set((0, 1)), set(values)

    previous_gen, = np.where(values == 1)

    yield previous_gen
    
    untagged_cells = set(np.where(values == 0)[0])

    while untagged_cells:
        # Collect vertices for computing
        # FIXME: not sure it is correct to only compute the neigbohors of
        # the previous layer, perhaps all should be included?
        vertices = np.unique(np.hstack(list(map(c2v, previous_gen))))
        # The new guys
        previous_gen = np.fromiter(untagged_cells & set(np.unique(np.hstack(list(map(v2c, vertices))))),
                                   dtype='uintp')

        yield previous_gen
        
        untagged_cells.difference_update(previous_gen)


def layer_neighbor_cells(tagged_cells, nlayers=5):
    '''Tag neighbors of tagged_cells by their distance from 1'''
    layers = df.MeshFunction('size_t',
                             tagged_cells.mesh(),
                             tagged_cells.mesh().topology().dim(),
                             0)
    values = layers.array()

    layer = layer_neighbor_cell_generator(tagged_cells)
    
    layer_tag = 1
    while layer_tag <= nlayers: 
        layer_cells = next(layer)
        values[layer_cells] = layer_tag
        layer_tag += 1
    return layers


def layer_neighbor_vertex_generator(edge_f, nlayers=5):
    '''Tag neighbors of tagged_cells by their distance from 1'''
    assert edge_f.dim() == 1
    assert set(edge_f.array()) == set((0, 1))

    mesh = edge_f.mesh()
    tdim = mesh.topology().dim()

    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    # First layer - vertices that make up marked edges
    layer0 = set(np.concatenate(list(map(e2v, np.where(edge_f.array() == 1)[0]))))
    yield layer0
    nlayers -= 1

    tagged_cells_idx = curve_isected_cells(edge_f, predicate=lambda x: x == 1)
    tagged_cells = df.MeshFunction('size_t', mesh, tdim, 0)
    tagged_cells.array()[tagged_cells_idx] = 1
    # Other ones shall be computed from cell neiboring
    mesh.init(tdim, 0)
    c2v = mesh.topology()(tdim, 0)
    # For peeling
    layers = layer_neighbor_cell_generator(tagged_cells)

    while nlayers:
        layer_cells = next(layers)
        layer_cells_as_vtx = set(np.concatenate(list(map(c2v, layer_cells))))
        # We keep those not connected to previous?
        layer0 = layer_cells_as_vtx - layer0

        yield layer0
        nlayers -= 1
        
        
def curve_distance(edge_f, nlayers=4, outside_val=-1):
    '''
    Compute distance (P1) function that has for each vertex distance 
    from curve edge_f == 1.
    '''
    # Want to build a P1 function
    mesh = edge_f.mesh()
    V = df.FunctionSpace(mesh, 'CG', 1)
    d = df.Function(V)
    d_values = d.vector().get_local()
    # Default
    d_values[:] = outside_val

    # Want to set points layer by layer
    v2d = df.vertex_to_dof_map(V)
    layers = layer_neighbor_vertex_generator(edge_f, nlayers=nlayers+1)

    curve_points = next(layers)
    # On curve is 0
    d_values[v2d[list(curve_points)]] = 0.


    x = mesh.coordinates()    
    # For others we need to compute distance from edges
    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    segments = np.row_stack(list(map(e2v, np.where(edge_f.array() == 1)[0])))
    # In local numbering
    vtx_idx, segments = np.unique(segments.flatten(), return_inverse=True)

    line_mesh = make_line_mesh(x[vtx_idx], segments.reshape((-1, 2)))
    tree = line_mesh.bounding_box_tree()

    for points in map(list, layers):
        d_values[v2d[points]] = np.fromiter((tree.compute_closest_entity(df.Point(x[p]))[1] for p in points),
                                            dtype=float)
    d.vector().set_local(d_values)

    return d
