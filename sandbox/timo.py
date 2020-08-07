# This requires perfusion package to be on path
from perfusion.vtp_read import read_vtp
from mbed.meshing import embed_mesh1d
import dolfin as df
import numpy as np
import sys


path = '../../perfusion/data/ratbrain.vtp'
# A cell function
f = read_vtp(path, (1, 'Radius [micron]'))

df.File('radius.pvd') << f

mesh1d = f.mesh()
df.File('time_mesh1d.xml') << mesh1d

original_coloring = df.MeshFunction('size_t', mesh1d, 1, 0)
original_coloring.array()[:] = np.arange(1, mesh1d.num_entities(1)+1)


# FIXME: can we embed something smaller?
#values = original_coloring.array()
#values[np.logical_and(values > 6000, values < 6100)] = 1
df.File('original.pvd') << original_coloring

#mesh1d = df.SubMesh(mesh1d, original_coloring, 1)
#original_coloring = df.MeshFunction('size_t', mesh1d, 1, 0)
#original_coloring.array()[:] = np.arange(1, mesh1d.num_entities(1)+1)
#df.File('original.pvd') << original_coloring


embedding = embed_mesh1d(mesh1d,
                         bounding_shape=0.01,
                         how='as_lines',
                         gmsh_args=list(sys.argv),
                         niters=12,                         
                         save_geo='model',
                         save_msh='model',
                         save_embedding='timo_rat')

# Counts of edges colorings?
# Embedding with edges
# Indentation
# Is line embedding?

coloring = embedding.edge_coloring
df.File('embedded.pvd') << coloring

# Who is missing -- figure this mapping out!
# coloring.array()[:] = 0
# original_coloring.array()[:] = 0
# mapping = embedding.nc_edge_encoding.as_edges
# # Only keep those that have been messed up
# for new_color, old_color in enumerate(mapping, 1):
#     coloring.array()[sum(mapping[old_color], [])] = new_color
#     original_coloring.array()[old_color-1] = new_color
    
# df.File('embedded_skewed.pvd') << coloring
# df.File('original_skewed.pvd') << original_coloring
