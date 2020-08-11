# This requires perfusion package to be on path
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

import matplotlib.pyplot as plt
from mbed.trimming import *
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
idx = find_edges(tmesh_radius, predicate=lambda v, x: v > 5.0)
rmesh, rcmap, rlmap = make_submesh(tmesh, idx)

df.File('rmesh.pvd') << rmesh

tagged_cc = connected_components(rmesh)
df.File('rmesh_components.pvd') << tagged_cc

# embedding = embed_mesh1d(mesh1d,
#                          bounding_shape=0.01,
#                          how='as_lines',
#                          gmsh_args=list(sys.argv),
#                          niters=12,                         
#                          save_geo='model',
#                          save_msh='model',
#                          save_embedding='timo_rat')

# # Counts of edges colorings?
# # Embedding with edges
# # Indentation
# # Is line embedding?

# coloring = embedding.edge_coloring
# df.File('embedded.pvd') << coloring

# # Who is missing -- figure this mapping out!
# # coloring.array()[:] = 0
# # original_coloring.array()[:] = 0
# # mapping = embedding.nc_edge_encoding.as_edges
# # # Only keep those that have been messed up
# # for new_color, old_color in enumerate(mapping, 1):
# #     coloring.array()[sum(mapping[old_color], [])] = new_color
# #     original_coloring.array()[old_color-1] = new_color
    
# # df.File('embedded_skewed.pvd') << coloring
# # df.File('original_skewed.pvd') << original_coloring
