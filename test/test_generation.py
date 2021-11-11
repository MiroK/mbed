from mbed.generation import make_line_mesh
import numpy as np
import ufl


def test_make_1():
    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1.]])
    mesh = make_line_mesh(coords, close_path=True)
    
    assert np.linalg.norm(mesh.coordinates() - coords) < 1E-13
    cells = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    
    assert np.linalg.norm(mesh.cells() - cells) < 1E-13, mesh.cells()


def test_make_2():
    coords = np.array([[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1., 2]])
    cells = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    
    mesh = make_line_mesh(coords, cells=cells)
    assert np.linalg.norm(mesh.coordinates() - coords) < 1E-13
    
    cells = np.array([[0, 1], [1, 2], [2, 3], [0, 3], [0, 2]])
    cells0 = np.array(list(map(sorted, mesh.cells())))
    
    assert np.linalg.norm(cells0 - cells) < 1E-13, mesh.cells()
    
    assert mesh.ufl_cell() == ufl.Cell('interval', 3)
