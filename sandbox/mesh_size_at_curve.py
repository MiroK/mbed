from mbed.curve_distance import curve_isected_cells
from itertools import combinations
import dolfin as df
import numpy as np
import os


folder = '/home/mirok/Downloads'
path = 'mesh.h5'

h5_file = os.path.join(folder, path)
embedding_mesh = df.Mesh()
h5 = df.HDF5File(embedding_mesh.mpi_comm(), h5_file, 'r')
h5.read(embedding_mesh, 'embedding_mesh', False)

edge_coloring = df.MeshFunction('size_t', embedding_mesh, 1, 0)
h5.read(edge_coloring, 'edge_coloring')

df.File('vasculature.pvd') << edge_coloring

isected_cells = df.MeshFunction('size_t', embedding_mesh, 3, 0)
# Get cells who have some vertices on 1d mesh
isected_cells.array()[curve_isected_cells(edge_coloring)] = 1

df.File('isected_cells.pvd') << isected_cells

# Utils for analysis
def volumes(mesh, cell_f=None):
    '''All volumes or those of cell_f == 1'''
    if cell_f is None:
        cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 1)

    for c in df.SubsetIterator(cell_f, 1):
        yield df.Cell(mesh, c.index()).volume()

                      
def edge_lengths(mesh, cell_f=None):
    '''All edge lengths or those of edge_f == 1'''
    if cell_f is None:
        cell_f = df.MeshFunction('size_t', mesh, mesh.topology().dim(), 1)

    mesh.init(mesh.topology().dim(), 0)
    x = mesh.coordinates()
    for c in df.SubsetIterator(cell_f, 1):
        vertices = c.entities(0)
        yield min(np.linalg.norm(x[v0]-x[v1]) for v0, v1 in combinations(vertices, 2))
        

vols = np.fromiter(volumes(embedding_mesh, isected_cells), dtype=float)
min_vol, mean_vol, max_vol = min(vols), np.mean(vols), max(vols)

lens = np.fromiter(edge_lengths(embedding_mesh, isected_cells), dtype=float)
min_el, mean_el, max_el = min(lens), np.mean(lens), max(lens)
