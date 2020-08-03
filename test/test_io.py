from mbed.utils import save_embedding, load_embedding
from mbed.meshing import embed_mesh1d

from test_meshing import _1d3d_mesh

import numpy as np
import os


def test_save():
    '''Not necesarily conform'''
    mesh1d = _1d3d_mesh(4)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             niters=1,
                             debug=False,
                             save_geo='')
                
    save_embedding(embedding, 'foo/bar')

    paths = [os.path.join('foo/bar', buz)
             for buz in ('mesh.h5', 'vertex_map.txt',
                         'edge_encoding_0.pkl', 'edge_encoding_1.pkl',
                         'nc_edge_encoding_0.pkl', 'nc_edge_encoding_1.pkl')]

    assert all(map(os.path.exists, paths))

    
def test_load():
    mesh1d = _1d3d_mesh(4)
    embedding = embed_mesh1d(mesh1d,
                             bounding_shape=0.1, 
                             how='as_points',
                             gmsh_args=[],
                             niters=1,
                             debug=False,
                             save_geo='')
                
    embedding0 = load_embedding(save_embedding(embedding, 'foo/bar'))

    assert embedding.edge_encoding == embedding0.edge_encoding
    assert embedding.nc_edge_encoding == embedding0.nc_edge_encoding
    assert np.linalg.norm(embedding.vertex_map - embedding0.vertex_map) < 1E-13
    assert np.linalg.norm(embedding.edge_coloring.array() - embedding0.edge_coloring.array()) < 1E-13
    assert np.linalg.norm(embedding.embedding_mesh.coordinates() -
                          embedding0.embedding_mesh.coordinates()) < 1E-13
    assert np.linalg.norm(embedding.embedding_mesh.cells() -
                          embedding0.embedding_mesh.cells()) < 1E-13
    
    
