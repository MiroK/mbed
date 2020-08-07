from itertools import combinations, product
from scipy.spatial.distance import pdist
from collections import defaultdict
from copy import deepcopy
import networkx as nx
import dolfin as df
import numpy as np


def get_terminal_map(mesh):
    '''Mapping from terminal nodes to edges connected to them'''
    assert mesh.topology().dim() == 1

    mesh.init(0, 1)
    v2e = mesh.topology()(0, 1)

    mapping = {}
    # A terminal node is such that 1 or 3 and more edges are connected to it
    for v in range(mesh.num_vertices()):
        es = v2e(v)
        assert len(es) > 0, 'Handing node %d' % v
        if len(es) == 1 or len(es) > 2:
            mapping[v] = set(es)

    return mapping


def get_branches(mesh, terminal_map=None):
    '''A branch goes from terminal to terminal. Return as vertex and edge encoding'''
    assert mesh.topology().dim() == 1
    
    if terminal_map is None:
        terminal_map = get_terminal_map(mesh)
    else:
        terminal_map = deepcopy(terminal_map)

    mesh.init(1, 0)
    mesh.init(0, 1)
    e2v, v2e = mesh.topology()(1, 0), mesh.topology()(0, 1)

    branches_e, branches_v = [], [] 
    for t in terminal_map:
        # Pick edge to start the branch
        while terminal_map[t]:
            edge = terminal_map[t].pop()

            branch_v, branch_e = [t], []
            # Walk until we reach another terminal
            reached_terminal = False
            vprev, vnext = t, -1
            while not reached_terminal:
                v0, v1 = e2v(edge)
                vnext = v0 if v1 == vprev else v1

                branch_v.append(vnext)
                branch_e.append(edge)
            
                reached_terminal = vnext in terminal_map
                if not reached_terminal:
                    edge, = set(v2e(vnext)) - set((edge, ))
                    vprev = vnext

            branches_v.append(branch_v)
            branches_e.append(branch_e)
        
            # Remove arrival edge from destination
            terminal_map[vnext] and terminal_map[vnext].remove(edge)

    return branches_e, branches_v


def branch_graph(branches):
    '''Nodes of the graph are branches'''
    v2b = defaultdict(set)
    # The idea here is to build map from vertex to branch(es)
    # that have it as terminal
    for bi, branch in enumerate(branches):
        v2b[branch[0]].add(bi)
        v2b[branch[-1]].add(bi)
    # Then it must be that there is a "edge" between branches that are mapped
    # to by the same node
    G = nx.Graph()
    for branches in v2b.values():
        G.add_edges_from(combinations(branches, 2))

    return G


def edge_lengths(mesh):
    '''P0 foo with edge lenths'''
    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)

    x = mesh.coordinates()
    lengths = np.linalg.norm(np.diff(ee, axis=1), 2, axis=2).T
    
    l = df.Function(df.FunctionSpace(mesh, 'DG', 0))
    l.vector().set_local(lengths)

    return l


def find_nodes(thing, predicate):
    '''Indices satisfying predicates'''
    if isinstance(thing, df.Mesh):
        return np.where(predicate(thing.coordinates()))[0]
    assert thing.dim() == 0
    # Mesh foo otherwise predicate mapd value and coordinates
    return np.where(predicate(thing.array(), thing.mesh().coordinates()))[0]


def find_edges(thing, predicate):
    '''Indices satisfying predicates'''
    if isinstance(thing, df.Mesh):
        # Edge midpoints
        x = thing.coordinates()
        thing.init(1, 0)
        e2v = thing.topology()(1, 0)
        mids = np.mean(x[map(e2v, range(thing.num_entities(1)))], axis=1)
        
        return np.where(predicate(mids))[0]
    
    assert thing.dim() == 0

    # Edge midpoints
    mesh = thing.mesh()
    x = mesh.coordinates()
    mesh.init(1, 0)
    e2v = mesh.topology()(1, 0)
    mids = np.mean(x[map(e2v, range(mesh.num_entities(1)))], axis=1)
    
    # Mesh foo otherwise
    return np.where(predicate(thing.array(), mids))[0]


def subwindows(mesh, dV):
    '''dV sized voxels that cover it'''
    x0 = mesh.coordinates().min(axis=0)
    x1 = mesh.coordinates().max(axis=0)
    if isinstance(dV, (int, float)):
        dV = (dV, )*len(x0)

    for x in product(*(np.arange(*t) for t in zip(x0, x1, dV))):
        yield x

# --------------------------------------------------------------------

from mbed.generation import make_line_mesh
import dolfin as df
import numpy as np

if True:
    mesh = df.Mesh('../sandbox/timo_mesh1d.xml')
else:
    x = np.array([[0., 0], [1, 1], [-1, 1], [-1, -1], [1, -1]])
    cells = np.array([[0, 1], [1, 2], [2, 0], [0, 3], [3, 4], [4, 0]])

    mesh = make_line_mesh(x, cells)

# print mesh.coordinates().min(axis=0), mesh.coordinates().max(axis=0)

x = mesh.coordinates()

dV = np.array([80, 80, 80])

f = df.MeshFunction('size_t', mesh, 1, 0)
values = f.array()

mesh.init(0, 1)
e2v, v2e = mesh.topology()(1, 0), mesh.topology()(0, 1)

tag = 1

terminals = get_terminal_map(mesh)
terminals = np.array([k for k, v in terminals.items() if len(v) > 2])

terminals_x = x[terminals]

distances = []
for x0 in subwindows(mesh, dV=dV):
    x1 = x0 + dV
    predicate = lambda x, x0=x0, x1=x1: np.all(np.logical_and(x > x0, x < x1), axis=1)
    idx, = np.where(predicate(terminals_x))
    
    if len(idx) > 7:
        D = pdist(terminals_x[idx])
    #print min(D), max(D), (min(idx), np.mean(idx), max(idx)), (x0, x1)
        d = np.mean(D)
        #distances.append(np.mean(D))
        if d < 30:
            print len(idx), d
            edges = np.unique(np.hstack(map(v2e, terminals[idx]))) 
            values[edges] = tag

            vertices = np.unique(np.hstack(map(e2v, edges)))
            print '\t', set(vertices) - set(terminals[idx])

            # We say the cluster center is the mean of terminal nodes
            # relative to the graph of the cluster, i.e. ignoring connections
            # outside

            # Does it make sense to talk about a ball?

            # We need to pick the terminals which will be reduced to some
            # point - it is perhaps safest to pick one of them instead of
            # creating a new point and risk having it collide with some
            # existing edge

            # If the clusters are isolated (assert this) then the updates
            # do not need modifs of mesh. Some bookeeping should be enough
            
            tag += 1
            
df.File('clusters_q.pvd') << f

#import matplotlib.pyplot as plt

#plt.figure()
#plt.hist(distances)
#plt.show()
    
    

#x0 = mesh.coordinates().min(axis=0)
#dx = mesh.coordinates().max(axis=1) - x0


# branch_as_e, branch_as_v = get_branches(mesh, terminal_map=None)
# print len(branch_as_e), mesh.num_cells()
# # Not duplicates

# # We got them all
# assert set(sum(branch_as_e, [])) == set(range(mesh.num_cells()))

# df.File('lengths.pvd') << edge_lengths(mesh)

# tagged_components = df.MeshFunction('size_t', mesh, 1, 0)
# values = tagged_components.array()

# dG = branch_graph(branch_as_v)
# ccs = sorted(nx.algorithms.connected_components(dG), key=len, reverse=True)
# for cc_tag, cc in enumerate(ccs[:1], 1):
#     # List of branches
#     cells = sum((branch_as_e[branch] for branch in cc), [])
#     values[cells] = cc_tag

# df.File('foo.pvd') << tagged_components


