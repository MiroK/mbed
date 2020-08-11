from mbed.generation import make_line_mesh
from mbed.preprocessing import refine
import numpy as np


def test():
    coords = np.array([[0, 0], [1, 0], [1, 0.2], [1, 0.5], [1, 0.7], [1, 1], [0, 1.]])
    mesh = make_line_mesh(coords, close_path=True)

    rmesh, mapping = refine(mesh, threshold=0.6)

    x = mesh.coordinates()
    y = rmesh.coordinates()
    assert np.linalg.norm(x - y[:len(x)]) < 1E-13

    e2v, E2V = mesh.topology()(1, 0), rmesh.topology()(1, 0)
    for c in range(mesh.num_cells()):
        x0, x1 = x[e2v(c)]
        e = x1 - x0
        e = e/np.linalg.norm(e)

        for C in np.where(mapping == c)[0]:
            y0, y1 = y[E2V(C)]
            E = y1 - y0
            E = E/np.linalg.norm(E)

            assert abs(1-abs(np.dot(e, E))) < 1E-13
