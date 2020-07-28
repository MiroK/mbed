from mbed.transferring import transfer_mesh_function
from mbed.meshing import embed_mesh1d
from mbed.utils import to_P1_function

from test_meshing import _1d2d_mesh, _1d3d_mesh

import dolfin as df
import numpy as np
import pytest


def _vertex_transfer(f, embedding):
    '''Use interpolation of archlength'''
    g = transfer_mesh_function(f, embedding)
    g = to_P1_function(g)

    if f.mesh().geometry().dim() == 2:
        g0 = df.interpolate(df.Expression('std::max(x[0], x[1])', degree=1), g.function_space())

        marks = embedding.edge_coloring
        marks.array()[marks.array() > 0] = 1

        dS = df.Measure('dS', domain=embedding.embedding_mesh, subdomain_data=marks)
        assert df.sqrt(abs(df.assemble(df.inner(df.avg(g0 - g), df.avg(g0 - g))*dS(1)))) < 1E-13
    else:
        # We're on line x=y=z
        g0 = df.interpolate(df.Expression('x[0]', degree=1), g.function_space())
        # And we'd like to do a line integral 
        marks = embedding.edge_coloring
        edges, = np.where(marks.array() > 0)  # The embedded ones

        f = lambda x: (g(x) - g0(x))**2

        x = embedding.embedding_mesh.coordinates()
        embedding.embedding_mesh.init(1, 0)        
        e2v = embedding.embedding_mesh.topology()(1, 0)
        
        result = 0.
        for edge in edges:
            x0, x1 = x[e2v(edge)]
            # Get quarature points for degree 2
            xq0 = 0.5*x0*(1-(-1./df.sqrt(3))) + 0.5*x1*(1+(-1./df.sqrt(3)))
            xq1 = 0.5*x0*(1-(1./df.sqrt(3))) + 0.5*x1*(1+(1./df.sqrt(3)))
            # Same reference weights equal to 1. So we just rescale
            wq = np.linalg.norm(x1 - x0)

            result += f(xq0)*wq + f(xq1)*wq
        result = df.sqrt(abs(result))

        assert result < 1E-13

    return True


def _edge_transfer(f, embedding):
    '''Check tag inheritance using edge coloring'''
    mapped_f = transfer_mesh_function(f, embedding).array()
    edge_colors = embedding.edge_coloring.array()
    for edge in range(f.size()):
        color = edge + 1
        mapped_edges, = np.where(edge_colors == color)
        assert np.all(~(mapped_f[mapped_edges] - f[edge]))
    return True


def test_line_2d():
    '''Not skew'''    
    mesh1d = _1d2d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_lines',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    f = df.MeshFunction('size_t', mesh1d, 1, 0)
    f.array()[:] = np.random.randint(1, 5, f.size())

    assert _edge_transfer(f, embedding)

    
def test_point_2d():
    '''Not skew'''    
    mesh1d = _1d2d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    f = df.MeshFunction('size_t', mesh1d, 1, 0)
    f.array()[:] = np.random.randint(1, 5, f.size())

    assert _edge_transfer(f, embedding)


def test_line_3d():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_lines',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    f = df.MeshFunction('size_t', mesh1d, 1, 0)
    f.array()[:] = np.random.randint(1, 5, f.size())

    assert _edge_transfer(f, embedding)

    
def test_point_3d():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    f = df.MeshFunction('size_t', mesh1d, 1, 0)
    f.array()[:] = np.random.randint(1, 5, f.size())

    assert _edge_transfer(f, embedding)

# ---

def test_line_2d_vertex():
    '''Not skew'''    
    mesh1d = _1d2d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_lines',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    f = df.MeshFunction('double', mesh1d, 0, 0)
    f.array()[:] = np.linalg.norm(mesh1d.coordinates(), np.inf, axis=1)

    assert _vertex_transfer(f, embedding)


def test_point_2d_vertex():
    '''Not skew'''    
    mesh1d = _1d2d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    f = df.MeshFunction('double', mesh1d, 0, 0)
    f.array()[:] = np.linalg.norm(mesh1d.coordinates(), np.inf, axis=1)

    assert _vertex_transfer(f, embedding)


def test_line_3d_vertex():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_lines',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')

    f = df.MeshFunction('double', mesh1d, 0, 0)
    f.array()[:] = mesh1d.coordinates()[:, 0]

    assert _vertex_transfer(f, embedding)


def test_point_3d_vertex():
    '''Not skew'''    
    mesh1d = _1d3d_mesh(3)
    embedding = embed_mesh1d(mesh1d,
                             padding=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             debug=False,
                             save_geo='')
    
    f = df.MeshFunction('double', mesh1d, 0, 0)
    f.array()[:] = mesh1d.coordinates()[:, 0]

    assert _vertex_transfer(f, embedding)
