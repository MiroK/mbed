from collections import namedtuple
import numpy as np
import dolfin as df


# We want to embed line mesh of Xd vertices `Gamma` into X-d hypercube
# This might require that the `Gamma` is modified by inserting auxiliary vertices.
# The result of embedding is a mesh `Omega`, edge function which encodes in the first
# len(Gamma.num_vertices) positions how vertices are embedded in Gamma.
# Edges of Gamma which correspond to edges of Gamma are encoded in edge_coloring
# which is edge function of Omega; color i >= 1 are edges which correspond to
# edge with index i-1 in Gamma. Vertex indices (in Omega numbering) making up
# (possibly **new**) edge (from refined segments) are in edge_encoding
LineMeshEmbedding = namedtuple('LineMeshEmebdding', ('embedding_mesh',
                                                     'vertex_map',
                                                     'edge_coloring',
                                                     'edge_encoding'))
# map[i].as_vertices is the i-th edge of the old mesh given as sequence of vertices
# in the embedding mesh
EdgeMap = namedtuple('EdgeMap', ('as_vertices', 'as_edges'))


def between(x, (x0, x1), tol):
    '''Approx x0 <= x <= x1'''
    assert x0 < x1
    return x >= x0 - tol and x <= x1 + tol


def is_on_line(p, A, B, tol=1E-12):
    '''Is p \in [A, B] segment'''
    # there is t in (0, 1) such that A + t*(B-A) = P
    t = np.dot(p-A, B-A)/np.linalg.norm(B-A)
    return between(t, (0, 1), tol)


def elmtype(iterable):
    '''Elementype of homogeneous container'''
    eltype, = set(map(type, iterable))
    return eltype


def insert(index, items, array):
    '''Multiinsert'''
    if isinstance(items, int):
        array.insert(index, items)
        return array
    # Iterable
    for index, item in enumerate(items, index):
        array.insert(index, item)
    return array


def to_P1_function(f):
    '''Vertex function -> P1'''
    assert f.dim() == 0, f.dim()

    V = df.FunctionSpace(f.mesh(), 'CG', 1)
    g = df.Function(V)
    g_values = g.vector().get_local()
    g_values[:] = f.array()[df.dof_to_vertex_map(V)]
    g.vector().set_local(g_values)

    return g


def hypercube(model, xmin, xmax, padding):
    '''Bounding box hypercube [x0, x1] will be increased by padding'''
    assert np.all(xmax > xmin)

    padding = padding*(xmax - xmin)

    xmin = xmin - 0.5*padding
    xmax = xmax + 0.5*padding

    if len(xmin) == 3:
        x0, y0, z0 = xmin
        x1, y1, z1 = xmax

        points = [(x0, y0, z0),
                  (x1, y0, z0),
                  (x1, y1, z0),
                  (x0, y1, z0),
                  (x0, y0, z1),
                  (x1, y0, z1),
                  (x1, y1, z1),
                  (x0, y1, z1)]

        # Add components to model
        points = np.array([model.geo.addPoint(*point) for point in points])
        lines = np.array([(1, 2), (2, 3), (3, 4), (4, 1),
                          (5, 6), (6, 7), (7, 8), (8, 5),
                          (1, 5), (2, 6), (3, 7), (4, 8)])
        # Above we have gmsh 1-based indexing
        lines = np.array([model.geo.addLine(*points[vs-1]) for vs in lines])

        surfs = np.array([(1, 2, 3, 4), (5, 6, 7, 8),
                          (1, 10, -5, -9), (2, 11, -6, -10),
                          (11, 7, -12, -3), (12, 8, -9, -4)])
        planes = []
        for surf in surfs:
            tag = model.geo.addCurveLoop(np.sign(surf)*lines[np.abs(surf)-1])
            planes.append(model.geo.addPlaneSurface([tag]))

        surface_loop = [model.geo.addSurfaceLoop(planes)]
        volumes = [model.geo.addVolume(surface_loop)]

        last_entity = {0: points[-1], 1: lines[-1], 2: planes[-1], 3: volumes[-1]}

        return last_entity

    assert len(xmin) == 2

    x0, y0 = xmin
    x1, y1 = xmax
    # NOTE: I assume here that in 2d we are in z = 0 plane
    points = [(x0, y0, 0),
              (x1, y0, 0),
              (x1, y1, 0),
              (x0, y1, 0)]

    points = np.array([model.geo.addPoint(*point) for point in points])

    lines = np.array([(1, 2), (2, 3), (3, 4), (4, 1)])
    lines = np.array([model.geo.addLine(*points[line-1]) for line in lines])   

    surfs = np.array([(1, 2, 3, 4)])
    planes = []
    for surf in surfs:
        tag = model.geo.addCurveLoop(np.sign(surf)*lines[np.abs(surf)-1])
        planes.append(model.geo.addPlaneSurface([tag]))

    last_entity = {0: points[-1], 1: lines[-1], 2: planes[-1]}

    return last_entity
