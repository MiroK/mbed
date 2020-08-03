from collections import defaultdict
import networkx as nx
import dolfin as df
import numpy as np
import conversion
import utils
import gmsh


def point_embed_mesh1d(model, mesh1d, bounding_shape, **kwargs):
    '''
    Embed points of mesh1d into Xd bounding shape. An attempt is made 
    to insert intermediate points so that also edges are embedded 
    '''
    x = mesh1d.coordinates()
    
    mesh1d.init(1, 0)
    e2v = mesh1d.topology()(1, 0)
    topology = [list(e2v(e)) for e in range(mesh1d.num_entities(1))]

    converged = False
    niters = kwargs.get('niters', 5)
    base_geo = kwargs['save_geo']
    for k in range(niters):
        # Some mesh which embeds points but where these points are not
        # necessarily edges
        if base_geo:
            kwargs['save_geo'] = '_'.join([base_geo, str(k)]) 
        t = utils.Timer('%d-th iteration of %d point embedding' % (k, len(x)))
        embedding_mesh, vmap = _embed_points(model, x, bounding_shape, **kwargs)
        t.done()
        
        assert _embeds_points(embedding_mesh, x, vmap)
        # See which edges need to be improved
        needs_embedding = _not_embedded_edges(topology, vmap, embedding_mesh)
        utils.print_green('# edges need embedding %d' % sum(map(len, needs_embedding)))
        converged = not any(needs_embedding)

        if kwargs['debug'] and k == niters - 1:
            gmsh.fltk.initialize()
            gmsh.fltk.run()

        if converged: break

        # Insert auxiliary points and retry
        t = utils.Timer('%d-th iteration of point insert' % k)        
        x, topology = _embed_edges(topology, x, needs_embedding)
        t.done()
        utils.print_green('# num points increased to %d' % len(x))

    skew_embed_vertex = defaultdict(list)
    # We capitulate and make approximations;    
    if not converged:
        utils.print_red('Falling back to non-conforming `embedding`')
        if base_geo:
            kwargs['save_geo'] = '_'.join([base_geo, str(niters)]) 
        
        embedding_mesh, vmap = _embed_points(model, x, bounding_shape, **kwargs)
        assert _embeds_points(embedding_mesh, x, vmap)

        needs_embedding = _not_embedded_edges(topology, vmap, embedding_mesh)
        # We "embed" the mesh using __only__ existing vertices - translate topology
        topology = [list(vmap[edge]) for edge in topology]
        # An edges that need embedding is a branch with terminal vertices - so the
        # idea is to insert the interior path vertices
        t = utils.Timer('Force embedding edges')
        topology = _force_embed_edges(topology, embedding_mesh, needs_embedding, skew_embed_vertex)
        t.done()
    else:
        # Since the original 1d mesh likely has been changed we give
        # topology wrt to node numbering of the embedding mesh
        topology = [list(vmap[edge]) for edge in topology]

    t = utils.Timer('Fishing for edges')
    # Need to color the edge function;
    embedding_mesh.init(1, 0)
    e2v = embedding_mesh.topology()(1, 0)
    edge_lookup = {tuple(sorted(e2v(e))): e for e in range(embedding_mesh.num_entities(1))}

    edge_f = df.MeshFunction('size_t', embedding_mesh, 1, 0)
    topology_as_edge = []
    for tag, edge in enumerate(topology, 1):
        the_edge = []
        for e in zip(edge[:-1], edge[1:]):
            edge_index = edge_lookup[tuple(sorted(e))]
            edge_f[edge_index] = tag
            the_edge.append(edge_index)
        topology_as_edge.append(the_edge)

    encode_edge = lambda path: [edge_lookup[tuple(sorted(e))] for e in zip(path[:-1], path[1:])]
    # Finally encode skew edges as edges
    skew_embed_edge = {k: map(encode_edge, edge_as_vertex)
                       for k, edge_as_vertex in skew_embed_vertex.items()}
    t.done()
    
    ans = utils.LineMeshEmbedding(embedding_mesh,
                                  # The others were not part of original data
                                  vmap[:mesh1d.num_vertices()],  
                                  edge_f,
                                  utils.EdgeMap(topology, topology_as_edge),
                                  utils.EdgeMap(skew_embed_vertex, skew_embed_edge))

    kwargs['save_embedding'] and utils.save_embedding(ans, kwargs['save_embedding'])


def _force_embed_edges(topology, mesh, edges2refine, skewed):
    '''
    We are in a situation where points cannot be joined by a segment so 
    we look for a shortest path
    '''
    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    edges = np.array(map(e2v, range(mesh.num_entities(1))))
    # For path computation build a graph where edges are weighted by
    # physical distance of nodes
    x = mesh.coordinates()
    edge_lengths = np.linalg.norm(x[edges[:, 0]] - x[edges[:, 1]], 2, axis=1)
    
    g = nx.Graph()
    for edge, weight in zip(edges, edge_lengths):
        g.add_edge(*edge, weight=weight)
    
    for index1d, (edge_vertices, slice_indices) in enumerate(zip(topology, edges2refine)):
        if not slice_indices: continue
        # We are going to insert new vertices which join
        # A C D E B-F
        #  0 1 2 3 4
        # A X Y C D E B F
        ninserted = 0
        for slice_index in slice_indices:
            first = slice_index + ninserted
            v0 = edge_vertices[first]
            v1 = edge_vertices[first + 1]
            # Shortest path of the graph
            path = nx.algorithms.shortest_path(g, v0, v1, weight='weight')
            # Insert
            utils.insert(first+1, path[1:-1], edge_vertices)
            ninserted += len(path) - 2

            skewed[index1d].append(list(path))
            
    return topology
            
        
    
def _embed_edges(topology, x, needs_embedding):
    '''Change topology and x by bisection edge that needs embedding'''
    new_x = []
    next_vertex = len(x)
    
    new_topology = []
    for edge, bisect_edges in zip(topology, needs_embedding):
        # bisect_edges is [] or [0, 1] ...
        if bisect_edges:
            nbisected = 0
            for edge_index in bisect_edges: # Prior to bisection
                v0, v1 = edge[edge_index+nbisected], edge[edge_index+nbisected+1]
                new_x.append(0.5*(x[v0] + x[v1]))
                
                edge.insert(edge_index + nbisected+1, next_vertex)
                
                nbisected += 1
                next_vertex += 1

    x = np.row_stack([x, np.row_stack(new_x)])
    
    return x, topology
            

def _not_embedded_edges(topology, vmap, mesh):
    '''Is there a straight connection between edges'''
    out = []

    star_of = {}

    mesh.init(1, 0)
    mesh.init(0, 1)
    e2v, v2e = mesh.topology()(1, 0), mesh.topology()(0, 1)

    compute_star = lambda v: set(np.hstack([e2v(ei) for ei in v2e(v)]))
    
    for edge in topology:
        edge_idx = []
        for i, (v0, v1) in enumerate(zip(edge[:-1], edge[1:])):
            v0, v1 = vmap[v0], vmap[v1]
            if v0 not in star_of:
                star_of[v0] = compute_star(v0)
            star0 = star_of[v0]

            if v1 not in star0:
                edge_idx.append(i)
        out.append(edge_idx)

    return out


def _embeds_points(mesh, x, vmap):
    '''mesh.x.vmap[i] == x[i]'''
    return max(np.linalg.norm(mesh.coordinates()[vmap] - x, 2, axis=1)) < 1E-13


def _embed_points(model, x, bounding_shape, **kwargs):
    '''Hypercube mesh with vertices x'''
    npoints, tdim = x.shape

    # Figure out how to bound it
    counts = bounding_shape.create_volume(model, x)
    
    # In gmsh Point(4) will be returned as fourth node
    vertex_map = []  # mesh_1d.x[i] is embedding_mesh[vertex_map[i]]
    for xi in x:
        vertex_map.append(model.geo.addPoint(*np.r_[xi, 0])-1)
    vertex_map = np.array(vertex_map)

    model.addPhysicalGroup(tdim, [counts[tdim]], 1)

    model.geo.synchronize()
    model.mesh.embed(0, 1+vertex_map, tdim, counts[tdim])
    model.geo.synchronize()

    kwargs['save_geo'] and gmsh.write('%s.geo_unrolled' % kwargs['save_geo'])

    model.mesh.generate(tdim)

    embedding_mesh, _ = conversion.mesh_from_gmshModel(model, include_mesh_functions=None)
    gmsh.clear()

    return embedding_mesh, vertex_map
