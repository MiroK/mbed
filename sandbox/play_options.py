from mbed.generation import make_line_mesh
from mbed.meshing import embed_mesh1d
import numpy as np
import sys


coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])
mesh1d = make_line_mesh(coords, close_path=True)

embed_mesh1d(mesh1d,
             bounding_shape=0.1,
             how='as_lines',
             gmsh_args=sys.argv,
             save_geo='model',
             save_msh='model',
             save_embedding='test_embed_line')

print()

embed_mesh1d(mesh1d,
             bounding_shape=0.1,
             how='as_points',
             gmsh_args=sys.argv,
             save_geo='model',
             save_msh='model',
             niters=2,
             save_embedding='test_embed_point')

