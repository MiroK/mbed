# This requires perfusion package to be on path
import matplotlib.pyplot as plt
from mbed.preprocessing import refine
from mbed.trimming import *
from perfusion.vtp_read import read_vtp
from mbed.meshing import embed_mesh1d
import dolfin as df
import numpy as np
import sys

path = '../../perfusion/data/ratbrain.vtp'
# A cell function
radius = read_vtp(path, (1, 'Radius [micron]'))

df.File('radius.pvd') << radius

mesh1d = radius.mesh()
df.File('timo_mesh1d.xml') << mesh1d

original_coloring = df.MeshFunction('size_t', mesh1d, 1, 0)
original_coloring.array()[:] = np.arange(1, mesh1d.num_entities(1)+1)

df.File('original.pvd') << original_coloring

# Getting thick enough vessels
radius_filter = lambda v, x: v > 4.5

# 0) Just filter on radius
if True:
    idx = find_edges(radius, predicate=radius_filter)
    rmesh, rcmap, rlmap = make_submesh(mesh1d, idx)

    df.File('meshr.pvd') << rmesh
    print edge_lengths(rmesh).vector().max(), rmesh.num_cells(), rmesh.num_vertices()

    rmesh, _ = refine(rmesh, threshold=20)
    print edge_lengths(rmesh).vector().max(), rmesh.num_cells(), rmesh.num_vertices()    

    embedding = embed_mesh1d(rmesh,
                         bounding_shape=0.01,
                         how='as_points',
                         gmsh_args=list(sys.argv),
                         save_geo='model',
                         save_msh='model',
                         save_embedding='point_rat_timo',
                         monitor=True,
                         niters=5)

    # This is a check that mesh is okay for solving    
    edge_f = embedding.edge_coloring
    df.File('point_rat_timo/embedded_r.pvd') << edge_f

    edge_f.array()[edge_f.array() > 0] = 1


    from perfusion.boundary_conditions import poisson_solve, EdgeDirichletBC
    
    V = df.FunctionSpace(embedding.embedding_mesh, 'CG', 1)
    f = df.Constant(1)
    bcs = EdgeDirichletBC(V, 2, edge_f, 1)
    
    _, uh = poisson_solve(V, f, bcs)

    df.File('point_rat_timo/poisson.pvd') << uh


exit()
# Preprocessing the mesh proceeds in several step.
# 1) Removing short edges
# Consider the curve below which plots for interval [l0, l1] a sum length
# of all segments of whose l0 < length < l1 divided by the total length
# of the vasculature. One can see that there is a gap in the probability
# between the smallest bin and the following ones. I interpret this as a
# noise and remove the "short" segments
bb, by_count, by_length = edge_histogram(mesh1d, nbins=100)

plt.figure()
plt.plot(0.5*(bb[:-1]+bb[1:]), by_length)
plt.show()

lengths = edge_lengths(mesh1d)
length_array = lengths.vector().get_local()
l0, l1 = length_array.min(), length_array.max()
l1 = l0 + 1*(l1-l0)/100.
tol = (l1 - l0)/1000.

idx = find_edges(lengths, predicate=lambda v, x: ~np.logical_and(l0 - tol < v,
                                                                 v < l1 + tol))
# This removal results in preserving almost the entire length of the mesh
print 'Reduced length', sum(length_array[idx])/sum(length_array)

lmesh, lcmap, lvmap = make_submesh(mesh1d, idx)
lengths = edge_lengths(lmesh)
length_array = lengths.vector().get_local()

df.File('lmesh.pvd') << lmesh

# 2) Topological filtering
# We consider the vasculature network as a graph and pick its largest connected component

tagged_cc = connected_components(lmesh)
idx = find_edges(tagged_cc, predicate=lambda v, x: v == 1)
# The reduction in area is not considerable; so the graph is indeed highly
# connected
print 'Reduced length', sum(length_array[idx])/sum(length_array)

tmesh, tcmap, tlmap = make_submesh(lmesh, idx)

df.File('tmesh.pvd') << tmesh

# 3) We want to include radius data effect
# Begin by transfering the data
tmesh_radius = df.MeshFunction('double', tmesh, 1, 0)
tmesh_radius.array()[:] = radius.array()[lcmap[tcmap]]

df.File('reduced_radius.pvd') << tmesh_radius

# Keep on cells with radius
idx = find_edges(tmesh_radius, predicate=radius_filter)
rmesh, rcmap, rlmap = make_submesh(tmesh, idx)

df.File('rmesh.pvd') << rmesh

tagged_cc = connected_components(rmesh, 'geometric')
df.File('rmesh_components.pvd') << tagged_cc

embedding = embed_mesh1d(rmesh,
                         bounding_shape=0.01,
                         how='as_points',
                         gmsh_args=list(sys.argv),
                         save_geo='model',
                         save_msh='model',
                         save_embedding='point_timo_rat',
                         monitor=True,
                         niters=5)

df.File('point_timo_rat/embedded_ltr.pvd') << embedding.edge_coloring

# This is a check that mesh is okay for solving    
edge_f = embedding.edge_coloring

edge_f.array()[edge_f.array() > 0] = 1


from perfusion.boundary_conditions import poisson_solve, EdgeDirichletBC
    
V = df.FunctionSpace(embedding.embedding_mesh, 'CG', 1)
f = df.Constant(1)
bcs = EdgeDirichletBC(V, 2, edge_f, 1)

_, uh = poisson_solve(V, f, bcs)

df.File('point_timo_rat/poisson.pvd') << uh