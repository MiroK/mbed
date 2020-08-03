from __future__ import print_function
from collections import namedtuple
import os, pickle, time
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
                                                     'edge_encoding',
                                                     'nc_edge_encoding'))
# map[i].as_vertices is the i-th edge of the old mesh given as sequence of vertices
# in the embedding mesh
EdgeMap = namedtuple('EdgeMap', ('as_vertices', 'as_edges'))
# Edges that are not embedded properly (in a sense that segment/cell in 1d
# is make of segments in embedding mesh which are not on the same line) are
# recorded in `nc_edge_encoding`


def save_embedding(embedding, folder):
    '''Save embedding
    folder/mesh.h5    <- embedding_mesh, edge_coloring
          /vertex_map.txt  <- numpy array of the vertex map
          /edge_encoding_0.pkl    pickles of the mappings
          /edge_encoding_1.pkl
          /nc_edge_encoding_0.pkl
          /nc_edge_encoding_1.pkl   
    '''
    not os.path.exists(folder) and os.makedirs(folder)
    # Mesh and coloring
    h5_file = os.path.join(folder, 'mesh.h5')
    out = df.HDF5File(embedding.embedding_mesh.mpi_comm(),
                      h5_file,
                      'w')
    out.write(embedding.embedding_mesh, 'embedding_mesh')
    out.write(embedding.edge_coloring, 'edge_coloring')

    # Vertex map
    np.savetxt(os.path.join(folder, 'vertex_map.txt'),
               embedding.vertex_map,
               header='Vertex map')

    # Edges
    with open(os.path.join(folder, 'edge_encoding_0.pkl'), 'wb') as out:
        pickle.dump(embedding.edge_encoding.as_vertices, out)

    with open(os.path.join(folder, 'edge_encoding_1.pkl'), 'wb') as out:
        pickle.dump(embedding.edge_encoding.as_edges, out)
        
    # Non-conforming edges
    with open(os.path.join(folder, 'nc_edge_encoding_0.pkl'), 'wb') as out:
        pickle.dump(embedding.nc_edge_encoding.as_vertices, out)

    with open(os.path.join(folder, 'nc_edge_encoding_1.pkl'), 'wb') as out:
        pickle.dump(embedding.nc_edge_encoding.as_edges, out)

    return folder


def load_embedding(folder):
    '''Load embedding'''
    h5_file = os.path.join(folder, 'mesh.h5')
    embedding_mesh = df.Mesh()
    h5 = df.HDF5File(embedding_mesh.mpi_comm(), h5_file, 'r')
    h5.read(embedding_mesh, 'embedding_mesh', False)
    
    edge_coloring = df.MeshFunction('size_t', embedding_mesh, 1, 0)
    h5.read(edge_coloring, 'edge_coloring')

    vertex_map = np.loadtxt(os.path.join(folder, 'vertex_map.txt'))

    edge_encoding = EdgeMap(pickle.load(open(os.path.join(folder, 'edge_encoding_0.pkl'), 'rb')),
                            pickle.load(open(os.path.join(folder, 'edge_encoding_1.pkl'), 'rb')))

    nc_edge_encoding = EdgeMap(pickle.load(open(os.path.join(folder, 'nc_edge_encoding_0.pkl'), 'rb')),
                               pickle.load(open(os.path.join(folder, 'nc_edge_encoding_1.pkl'), 'rb')))

    return LineMeshEmbedding(embedding_mesh,
                             vertex_map,
                             edge_coloring,
                             edge_encoding,
                             nc_edge_encoding)


def is_number(num):
    '''Number type'''
    return isinstance(num, (int, float, np.number))

                      
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


def hypercube(model, xmin, xmax):
    '''Bounding box hypercube [x0, x1]'''
    assert np.all(xmax > xmin)

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


def print_red(value, *values):
    RED = '\033[1;37;31m%s\033[0m'
    values = (value, ) + values    
    print(*[RED % v for v in values], sep=' ')


def print_green(value, *values):
    GREEN = '\033[1;37;32m%s\033[0m'
    values = (value, ) + values    
    print(*[GREEN % v for v in values], sep=' ')


def print_blue(value, *values):
    BLUE = '\033[1;37;34m%s\033[0m'
    values = (value, ) + values
    print(*[BLUE % v for v in values], sep=' ')


class Timer(object):
    def __init__(self, message, parent=None):
        self.message = message
        self.indent = 0 if parent is None else 1+parent.indent
        self.t0 = time.time()
        print_blue('\t'*self.indent, 'Starting', self.message)

    def done(self):
        print_blue('\t'*self.indent, 'Finnished', self.message, 'in %g seconds' % (time.time() - self.t0))
