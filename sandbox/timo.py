# This requires perfusion package to be on path
from perfusion.vtp_read import read_vtp
from mbed.meshing import embed_mesh1d
import dolfin as df
import numpy as np
import sys


path = '../../perfusion/data/ratbrain.vtp'
# A cell function
f = read_vtp(path, (1, 'Radius [micron]'))

mesh1d = f.mesh()
original_coloring = df.MeshFunction('size_t', mesh1d, 1, 0)
original_coloring.array()[:] = np.arange(1, mesh1d.num_entities(1)+1)

df.File('original.pvd') << original_coloring

embedding = embed_mesh1d(mesh1d,
                         bounding_shape=0.01,
                         how='as_points',
                         gmsh_args=sys.argv,
                         save_geo='model',
                         niters=6,
                         save_embdding='timo_rat')

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
