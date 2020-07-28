import numpy as np
import conversion
import utils
import gmsh


def line_embed_mesh1d(model, mesh1d, padding, **kwargs):
    '''Embed mesh1d in Xd square mesh'''
    x0 = mesh1d.coordinates().min(axis=0)
    x1 = mesh1d.coordinates().max(axis=0)

    tdim = len(x0)
    # Figure out how to bound it
    counts = utils.hypercube(model, x0, x1, padding)
    
    # In gmsh Point(4) will be returned as fourth node
    vertex_map = []  # mesh_1d.x[i] is embedding_mesh[vertex_map[i]]
    for xi in mesh1d.coordinates():
        vertex_map.append(model.geo.addPoint(*np.r_[xi, 0])-1)
    vertex_map = np.array(vertex_map)

    # Add lines of 1d
    mesh1d.init(1, 0)
    e2v = mesh1d.topology()(1, 0)
    lines, edge_encoding = [], []
    for edge in range(mesh1d.num_entities(1)):
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

    if kwargs.get('debug', False):
        gmsh.fltk.initialize()
        gmsh.fltk.run()

    if kwargs.get('save_geo', ''):
        gmsh.write('%s.geo_unrolled' % kwargs.get('save_geo'))

    model.mesh.generate(tdim)
    embedding_mesh, mesh_fs = conversion.mesh_from_gmshModel(model,
                                                             include_mesh_functions=1)
    
    gmsh.clear()

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
    # Combine
    edge_encoding = meshing.EdgeMap(edge_encoding, topology_as_edge)    
                
    return meshing.LineMeshEmbedding(embedding_mesh, vertex_map, edge_f, edge_encoding)
