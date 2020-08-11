from mbed.meshing import embed_mesh1d
from mbed.utils import is_on_line
import dolfin as df
import numpy as np
import pytest


def _embed_vertex(line_mesh, embedding_res):
    '''Vertices are embedded correctly'''
    omega = embedding_res.embedding_mesh
    Y = omega.coordinates()[embedding_res.vertex_map]
    x = line_mesh.coordinates()

    return max(np.linalg.norm(x - Y, 2, axis=1)) < 1E-13


def _embed_edgecolor(line_mesh, embedding_res):
    '''Picking vertices by edge color we get those declared'''
    array = embedding_res.edge_coloring.array()

    emesh = embedding_res.embedding_mesh
    emesh.init(0, 1)
    v2e = emesh.topology()(0, 1)
    for tag, edge in enumerate(embedding_res.edge_encoding.as_vertices, 1):
        edges, = np.where(array == tag)
        assert len(edges) == len(edge) - 1
        # We can find the edge and it has the color
        edges = set(edges)
        for v0, v1 in zip(edge[:-1], edge[1:]):
            e, = set(v2e(v0)) & set(v2e(v1))
            assert e in edges

    return True


def _embed_edgeencode(line_mesh, embedding_res):
    '''Vertices on edge are really there'''
    x = embedding_res.embedding_mesh.coordinates()
    
    line_mesh.init(1, 0)
    e2v = line_mesh.topology()(1, 0)
    for edge_idx, edge in enumerate(embedding_res.edge_encoding.as_vertices):
        # In embedding_mesh numbering        
        v0, v1 = edge[0], edge[-1]  
        # Grab it from line_mesh
        V0, V1 = embedding_res.vertex_map[e2v(edge_idx)]

        assert v0 == V0 and v1 == v1
        # The points are indeed on the edge
        edge_x = x[edge]
        A, B = edge_x[0], edge_x[-1]
        assert all(is_on_line(P, A, B) for P in edge_x)

    return True


def _1d2d_mesh(n, m=None):
    '''Line mesh in 2d'''
    if m is None: m = n
    mesh1d = df.UnitSquareMesh(n, m)
    mesh1d = df.BoundaryMesh(mesh1d, 'exterior')

    cell_f = df.MeshFunction('size_t', mesh1d, 1, 0)
    df.CompiledSubDomain('near(x[0], 0) || near(x[1], 0)').mark(cell_f, 1)
    mesh1d = df.SubMesh(mesh1d, cell_f, 1)

    return mesh1d


def _1d3d_mesh(n):
    '''Line mesh in 3d'''
    mesh1d = df.UnitCubeMesh(n, n, n)

    cell_f = df.MeshFunction('size_t', mesh1d, 3, 0)
    df.CompiledSubDomain('x[0] > x[1] - DOLFIN_EPS').mark(cell_f, 1)#
    mesh1d = df.BoundaryMesh(df.SubMesh(mesh1d, cell_f, 1), 'exterior')
    
    cell_f = df.MeshFunction('size_t', mesh1d, 2, 0)
    df.CompiledSubDomain('near(x[0], x[1])').mark(cell_f, 1)
    mesh1d = df.SubMesh(mesh1d, cell_f, 1)

    cell_f = df.MeshFunction('size_t', mesh1d, 2, 0)
    df.CompiledSubDomain('x[2] < x[1] + DOLFIN_EPS').mark(cell_f, 1)
    mesh1d = df.BoundaryMesh(df.SubMesh(mesh1d, cell_f, 1), 'exterior')
    
    cell_f = df.MeshFunction('size_t', mesh1d, 1, 0)
    df.CompiledSubDomain('near(x[2], x[1])').mark(cell_f, 1)
    mesh1d = df.SubMesh(mesh1d, cell_f, 1)


    return mesh1d


def test_line_2d():
    '''Not skew'''    
    mesh1d = _1d2d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_lines',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    # assert not status
    assert _embed_vertex(mesh1d, embedding)
    assert _embed_edgecolor(mesh1d, embedding)
    assert _embed_edgeencode(mesh1d, embedding)


def test_point_2d():
    '''Not skew'''
    mesh1d = _1d2d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    # assert not status
    assert _embed_vertex(mesh1d, embedding)
    assert _embed_edgecolor(mesh1d, embedding)
    assert _embed_edgeencode(mesh1d, embedding)


def test_line_3d():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_lines',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    # assert not status
    assert _embed_vertex(mesh1d, embedding)
    assert _embed_edgecolor(mesh1d, embedding)
    assert _embed_edgeencode(mesh1d, embedding)


def test_point_3d():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,           # dolfin.Mesh
                             bounding_shape=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')
    
    # assert not status
    assert _embed_vertex(mesh1d, embedding)
    assert _embed_edgecolor(mesh1d, embedding)
    assert _embed_edgeencode(mesh1d, embedding)
    

def test_point_skew_2d():
    '''Not necesarily conform'''
    mesh1d = _1d2d_mesh(4)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             niters=1,
                             debug=False,
                             save_geo='')

    skewed = embedding.nc_edge_encoding.as_vertices
    assert skewed

    x = embedding.embedding_mesh.coordinates()

    embedding.embedding_mesh.init(1, 0)
    embedding.embedding_mesh.init(0, 1)
    E2V, V2E = embedding.embedding_mesh.topology()(1, 0), embedding.embedding_mesh.topology()(0, 1)

    compute_star = lambda v: set(np.hstack([E2V(ei) for ei in V2E(v)]))

    y = mesh1d.coordinates()
    e2v = mesh1d.topology()(1, 0)
    for edge in skewed:
        y0, y1 = y[e2v(edge)]
        for piece in skewed[edge]:
            if not len(piece) == 3:
                continue
            x0, x2, x1 = x[piece]
            # End points were inserted correctly
            assert is_on_line(x0, y0, y1)

            v0, v2, v1 = piece
            # Mid point is in the star
            mids = compute_star(v0) & compute_star(v1)

            assert v2 in mids
            # If there are more this way we get a shortest path
            if len(mids) > 1:
                path_length = lambda v: np.linalg.norm(x[v0] - x[v], 2) + np.linalg.norm(x[v1] - x[v], 2) 
                assert v2 == min(mids, key=path_length)

    vmap = embedding.vertex_map
    encode = embedding.edge_encoding.as_edges
    print encode
    print skewed
    # We can combine the maps to pick correctly embedded edges
    for edge in range(mesh1d.num_cells()):
        if edge not in skewed:
            v0, v1 = e2v(edge)

            e_ = encode[edge]
            print e_, '<--'
            
            vs = set(vmap[e2v(e_[0])])
            assert v0 not in vs

            vs = set(vmap[e2v(e_[-1])])
            assert v1 in vs

            
def test_point_skewPartly_2d():
    '''Not necesarily conform'''
    mesh1d = _1d2d_mesh(4, 32)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             niters=1,
                             debug=False,
                             save_geo='')

    skewed = embedding.nc_edge_encoding.as_vertices
    assert skewed

    x = embedding.embedding_mesh.coordinates()

    embedding.embedding_mesh.init(1, 0)
    embedding.embedding_mesh.init(0, 1)
    E2V, V2E = embedding.embedding_mesh.topology()(1, 0), embedding.embedding_mesh.topology()(0, 1)

    compute_star = lambda v: set(np.hstack([E2V(ei) for ei in V2E(v)]))

    y = mesh1d.coordinates()
    e2v = mesh1d.topology()(1, 0)
    for edge in skewed:
        y0, y1 = y[e2v(edge)]
        for piece in skewed[edge]:
            if not len(piece) == 3:
                continue
            x0, x2, x1 = x[piece]
            # End points were inserted correctly
            assert is_on_line(x0, y0, y1)

            v0, v2, v1 = piece
            # Mid point is in the star
            mids = compute_star(v0) & compute_star(v1)

            assert v2 in mids
            # If there are more this way we get a shortest path
            if len(mids) > 1:
                path_length = lambda v: np.linalg.norm(x[v0] - x[v], 2) + np.linalg.norm(x[v1] - x[v], 2) 
                assert v2 == min(mids, key=path_length)

    vmap = embedding.vertex_map
    encode = embedding.edge_encoding.as_edges
    # We can combine the maps to pick correctly embedded edges
    for edge in range(mesh1d.num_cells()):
        if edge not in skewed:
            # Look up edge in embedding numbering
            vs0 = set(vmap[e2v(edge)])
            E_,  = encode[edge]
            # Corresponding encoded
            vs = set(E2V(E_))

            assert vs == vs0

 
def test_point_skew_3d():
    '''Not necesarily conform'''
    mesh1d = _1d3d_mesh(4)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             niters=1,
                             debug=False,
                             save_geo='')

    skewed = embedding.nc_edge_encoding.as_vertices
    assert skewed

    x = embedding.embedding_mesh.coordinates()

    embedding.embedding_mesh.init(1, 0)
    embedding.embedding_mesh.init(0, 1)
    E2V, V2E = embedding.embedding_mesh.topology()(1, 0), embedding.embedding_mesh.topology()(0, 1)

    compute_star = lambda v: set(np.hstack([E2V(ei) for ei in V2E(v)]))

    y = mesh1d.coordinates()
    e2v = mesh1d.topology()(1, 0)
    for edge in skewed:
        y0, y1 = y[e2v(edge)]
        for piece in skewed[edge]:
            assert len(piece) == 3
            x0, x2, x1 = x[piece]
            # End points were inserted correctly
            assert is_on_line(x0, y0, y1)

            v0, v2, v1 = piece
            # Mid point is in the star
            mids = compute_star(v0) & compute_star(v1)

            assert v2 in mids
            # If there are more this way we get a shortest path
            if len(mids) > 1:
                path_length = lambda v: np.linalg.norm(x[v0] - x[v], 2) + np.linalg.norm(x[v1] - x[v], 2) 
                assert v2 == min(mids, key=path_length)


def test_line_stl_3d():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape='box.stl', 
                             how='as_lines',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    # assert not status
    assert _embed_vertex(mesh1d, embedding)
    assert _embed_edgecolor(mesh1d, embedding)
    assert _embed_edgeencode(mesh1d, embedding)


def test_point_stl_3d():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape='box.stl', 
                             how='as_points',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    # assert not status
    assert _embed_vertex(mesh1d, embedding)
    assert _embed_edgecolor(mesh1d, embedding)
    assert _embed_edgeencode(mesh1d, embedding)


def test_point_skew_stl_3d():
    '''Not necesarily conform'''
    mesh1d = _1d3d_mesh(4)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape='box.stl', 
                             how='as_points',
                             gmsh_args=[],
                             niters=1,
                             debug=False,
                             save_geo='')

    skewed = embedding.nc_edge_encoding.as_vertices
    assert skewed

    x = embedding.embedding_mesh.coordinates()

    embedding.embedding_mesh.init(1, 0)
    embedding.embedding_mesh.init(0, 1)
    E2V, V2E = embedding.embedding_mesh.topology()(1, 0), embedding.embedding_mesh.topology()(0, 1)

    compute_star = lambda v: set(np.hstack([E2V(ei) for ei in V2E(v)]))

    y = mesh1d.coordinates()
    e2v = mesh1d.topology()(1, 0)
    for edge in skewed:
        y0, y1 = y[e2v(edge)]
        for piece in skewed[edge]:
            assert len(piece) == 3
            x0, x2, x1 = x[piece]
            # End points were inserted correctly
            assert is_on_line(x0, y0, y1)

            v0, v2, v1 = piece
            # Mid point is in the star
            mids = compute_star(v0) & compute_star(v1)

            assert v2 in mids
            # If there are more this way we get a shortest path
            if len(mids) > 1:
                path_length = lambda v: np.linalg.norm(x[v0] - x[v], 2) + np.linalg.norm(x[v1] - x[v], 2) 
                assert v2 == min(mids, key=path_length)


test_point_skewPartly_2d()                
