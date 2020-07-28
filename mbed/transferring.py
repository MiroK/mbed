import numpy as np


def transfer_mesh_function(f, embedding):
    '''Transfer edge or vertex function from line mesh'''
    mesh = f.mesh()

    return {0: _transfer_vertex_function,
            1: _transfer_edge_function}[f.dim()](f, embedding)


def _transfer_edge_function(f, embedding):
    '''Child edges inherit "tag" of the parent'''
    g = type(f)(embedding.embedding_mesh, 1, 0)
    g_values = g.array()

    f_values = f.array()
    mapping = embedding.edge_encoding.as_edges
    for edge, mapped_edge in enumerate(mapping):
        g_values[mapped_edge] = f_values[edge]
    return g


def _transfer_vertex_function(f, embedding):
    '''Child vertex value by linear interpolation (via arch length coordinate)'''
    f_values = f.array()
    mapping = embedding.edge_encoding.as_vertices


    f.mesh().init(1, 0)
    e2v = f.mesh().topology()(1, 0)
    vmap = embedding.vertex_map

    x = embedding.embedding_mesh.coordinates()
    g = type(f)(embedding.embedding_mesh, 0, 0)
    g_values = g.array()    
    for edge, mapped_edge in enumerate(mapping):
        v0, v1 = e2v(edge)
        assert vmap[v0] == mapped_edge[0], ((v0, v1), (mapped_edge[0], mapped_edge[-1]))
        assert vmap[v1] == mapped_edge[-1]

        f0, f1 = f_values[v0], f_values[v1]
        if len(mapped_edge) == 2:
            g_values[mapped_edge] = np.array([f0, f1])
        else:
            lengths = [np.linalg.norm(x[v0]-x[v1]) for v0, v1 in zip(mapped_edge[:-1], mapped_edge[1:])]
            # A + (B-A)*t  for t in (0, 1)
            arc_lengths = np.r_[0, np.cumsum(lengths)]
            arc_lengths /= arc_lengths[-1]
            # The rest is interpolation
            g_values[mapped_edge] = f0 + arc_lengths*(f1 - f0)
    return g
