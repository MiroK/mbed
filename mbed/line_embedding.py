import numpy as np
import conversion
import utils
import tqdm
import gmsh
import os


def line_embed_mesh1d(model, mesh1d, bounding_shape, **kwargs):
    '''Embed mesh1d in Xd square mesh'''
    time_model = utils.Timer('Line embedding model definition')
    npoints, tdim = mesh1d.coordinates().shape
    # Figure out how to bound it
    counts = bounding_shape.create_volume(model, mesh1d.coordinates())
    
    # In gmsh Point(4) will be returned as fourth node
    vertex_map = []  # mesh_1d.x[i] is embedding_mesh[vertex_map[i]]
    for xi in mesh1d.coordinates():
        vertex_map.append(model.geo.addPoint(*np.r_[xi, 0])-1)
    vertex_map = np.array(vertex_map)

    # Add lines of 1d
    mesh1d.init(1, 0)
    e2v = mesh1d.topology()(1, 0)
    lines, edge_encoding = [], []
    for edge in tqdm.tqdm(range(mesh1d.num_entities(1))):
        v0, v1 = vertex_map[e2v(edge)] + 1
        line = model.geo.addLine(v0, v1)
        # There will be a edge function such that edge corresponding
        # to edge `i` in mesh1d will have tag `i`
        model.addPhysicalGroup(1, [line], edge+1)
        lines.append(line)

        edge_encoding.append([v0-1, v1-1])

    model.addPhysicalGroup(tdim, [counts[tdim]], 1)

    model.geo.synchronize()
    model.mesh.embed(1, lines, tdim, counts[tdim])
    model.geo.synchronize()
    # --
    time_model.done()
    
    if kwargs['debug']:
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    kwargs['save_geo'] and gmsh.write('%s.geo_unrolled' % kwargs['save_geo'])

    time_gen = utils.Timer('Generation line embedded mesh', 1)
    model.mesh.generate(tdim)
    time_gen.done()

    time_conv = utils.Timer('Mesh conversion', 1)
    embedding_mesh, mesh_fs = conversion.mesh_from_gmshModel(model,
                                                             include_mesh_functions=1)
    time_conv.done()
    
    gmsh.clear()

    time_edge_encode = utils.Timer('Fishing for embedded edges', 1)    
    edge_f = mesh_fs[1]
    edge_values =  edge_f.array()

    embedding_mesh.init(1, 0)
    e2v = embedding_mesh.topology()(1, 0)
    x = embedding_mesh.coordinates()    
    # It remains to account for the nodes that might have been inserted
    # on the edge
    topology_as_edge = []
    for tag, edge in enumerate(edge_encoding, 1):
        edges, = np.where(edge_values == tag)
        topology_as_edge.append(list(edges))
        if len(edges) > 1:
            nodes = np.unique(np.hstack([e2v(e) for e in edges]))
            # NOTE: Here we use the fact that we have a straight line so
            # we simply order interior nodes of the edge by their distance
            # from start
            idx = np.argsort(np.linalg.norm(x[nodes] - x[edge[0]], 2, axis=1))
            nodes = nodes[idx]
            assert nodes[-1] == edge[1]
            # Insder them<
            for i, n in enumerate(nodes[1:-1], 1):
                edge.insert(i, n)
    time_edge_encode.done()
    
    # Combine
    edge_encoding = utils.EdgeMap(edge_encoding, topology_as_edge)    
    skew_encoding = utils.EdgeMap({}, {})

    ans = utils.LineMeshEmbedding(embedding_mesh, vertex_map, edge_f, edge_encoding, skew_encoding)
    
    kwargs['save_embedding'] and utils.save_embedding(ans, kwargs['save_embedding'])

    return ans

