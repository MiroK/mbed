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
    x = mesh.coordinates()
    lengths = np.linalg.norm(np.diff(x[mesh.cells()], axis=1), 2, axis=2).reshape((-1, ))
    
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

    if isinstance(thing, df.Function):
        mesh = thing.function_space().mesh()
        f = df.MeshFunction('double', mesh, mesh.topology().dim(), 0.)
        f.array()[:] = thing.vector().get_local()

        return find_edges(f, predicate)

    assert thing.dim() == 1

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


def edge_histogram(thing, nbins, tol=0.01):
    '''Bins and histogram by edge count and edge length'''
    if isinstance(thing, df.Mesh):
        return edge_histogram(edge_lengths(thing), nbins)

    if isinstance(thing, (df.MeshFunction, df.MeshFunctionBool, df.MeshFunctionSizet,
                          df.MeshFunctionInt, df.MeshFunctionDouble)):
        f = df.Function(df.FunctionSpace(thing.mesh(), 'DG', 0))
        f.vector().set_local(np.fromiter(thing.array(), dtype=float))
        return edge_histogram(f, nbins)

    assert thing.function_space().mesh().topology().dim() == 1
    
    values = thing.vector().get_local()
    vmin, vmax = values.min(), values.max()
    # For edge length weighting
    lengths = edge_lengths(thing.function_space().mesh()).vector().get_local()
    tol = (tol/nbins)*(vmax - vmin)
    # Normalization
    total_count = float(len(values))
    total_length = sum(lengths)
    
    bins = np.linspace(vmin, vmax, nbins)
    
    bin_c_values, bin_values = np.zeros(len(bins)-1), np.zeros(len(bins)-1)
    for i, (l0, l1) in enumerate(zip(bins[:-1], bins[1:])):
        idx, = np.where(np.logical_and(l0 - tol < values, values < l1 + tol))
        if len(idx):
            print l0-tol, l1+tol, len(idx)
            bin_c_values[i] = len(idx)/total_count
            bin_values[i] = sum(lengths[idx])/total_length

    return bins, bin_c_values, bin_values


def connected_components(mesh):
    '''A mesh function which colors connected components of 1d graph'''
    # Largest components have smallest tag
    tagged_components = df.MeshFunction('size_t', mesh, 1, 0)
    values = tagged_components.array()

    branch_as_e, branch_as_v = get_branches(mesh, terminal_map=None)
    dG = branch_graph(branch_as_v)
    ccs = sorted(nx.algorithms.connected_components(dG), key=len, reverse=True)

    for cc_tag, cc in enumerate(ccs[:1], 1):
        # List of branches
        cells = sum((branch_as_e[branch] for branch in cc), [])
        values[cells] = cc_tag

    return tagged_components


def make_submesh(mesh, use_indices):
    '''Submesh + mapping of child to parent cell and vertex indices'''
    tdim = mesh.topology().dim()

    f = df.MeshFunction('size_t', mesh, tdim, 0)
    f.array()[use_indices] = 1

    submesh = df.SubMesh(mesh, f, 1)
    return (submesh,
            submesh.data().array('parent_cell_indices', tdim),
            submesh.data().array('parent_vertex_indices', 0))

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


lengths = edge_lengths(mesh)
length_array = lengths.vector().get_local()
l0, l1 = length_array.min(), length_array.max()
l1 = l0 + 2*(l1-l0)/100.
tol = (l1 - l0)/1000.

idx = find_edges(lengths, predicate=lambda v, x: ~np.logical_and(l0 - tol < v,
                                                                 v < l1 + tol))
print 'Reduced length by', 1-sum(length_array[idx])/sum(length_array)

# ---

lmesh, _, __ = make_submesh(mesh, idx)
lengths = edge_lengths(lmesh)
length_array = lengths.vector().get_local()

tagged_cc = connected_components(lmesh)
idx = find_edges(tagged_cc, predicate=lambda v, x: v == 1)

print 'Reduced length by', 1-sum(length_array[idx])/sum(length_array)

tmesh, _, __ = make_submesh(lmesh, idx)

df.File('foo.pvd') << tmesh


x = tmesh.coordinates()

dV = np.array([40, 40, 40])

f = df.MeshFunction('size_t', tmesh, 1, 0)
values = f.array()

tmesh.init(0, 1)
e2v, v2e = tmesh.topology()(1, 0), tmesh.topology()(0, 1)

tag = 1

terminal_map = get_terminal_map(tmesh)
terminals = np.array([k for k, v in terminal_map.items() if len(v) > 2])

terminals_x = x[terminals]

cell_nodes = tmesh.cells().flatten()

kill_node = tmesh.num_vertices() + 1
for x0 in subwindows(tmesh, dV=dV):
    x1 = x0 + dV
    predicate = lambda x, x0=x0, x1=x1: np.all(np.logical_and(x > x0, x < x1), axis=1)
    idx, = np.where(predicate(terminals_x))
    
    if len(idx) > 6:
        cluster_points = terminals[idx]
        cluster_x = terminals_x[idx] 
        D = pdist(cluster_x)
    #print min(D), max(D), (min(idx), np.mean(idx), max(idx)), (x0, x1)
        d = np.mean(D)
        #distances.append(np.mean(D))
        if d < 50:
            edges = np.unique(np.hstack(map(v2e, cluster_points))) 
            values[edges] = tag

            vertices = np.unique(np.hstack(map(e2v, edges)))

            # center = np.mean(cluster_x, axis=0)
            # center = cluster_points[np.argmin(np.linalg.norm(cluster_x - center, 2, axis=1))]

            for p in cluster_points:
                cell_nodes[cell_nodes == p] = kill_node #center
            
            #subs, = np.where(cell_nodes[0::2] == cell_nodes[1::2])
            #my_subs = set(subs) - invalid
            #print len(my_subs)
            #invalid.update(my_subs)

            tag += 1
# print('Cluster count', tag)

# cells = cell_nodes.reshape((-1, 2))

# # invalid, = np.where(cells[:, 0] == cells[:, 1])
# invalid = np.unique(np.r_[np.where(cells[:, 0] == kill_node)[0],
#                           np.where(cells[:, 1] == kill_node)[0]])

# print len(invalid), len(cells)

# cells = np.delete(cells, invalid, axis=0)

# # Recombine
# nodes = np.unique(cells.flatten())
# old2new = {o: n for n, o in enumerate(nodes)}

# new_x = x[nodes]
# new_cells = np.array([old2new[n] for n in cells.flatten()]).reshape((-1, 2))

# from mbed.generation import make_line_mesh

# xx = make_line_mesh(new_x, new_cells)

# df.File('lala.pvd') << xx

            
df.File('clusters_q.pvd') << f
