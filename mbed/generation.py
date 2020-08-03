import dolfin as df
import numpy as np


def make_line_mesh(vertices, cells=None, close_path=False):
    '''Mesh from data by MeshEditor'''
    assert vertices.ndim == 2
    # Path mesh
    if cells is None:
        nvertices = len(vertices)
        
        if not close_path:
            cells = np.array([(i, i+1) for i in range(nvertices-1)], dtype='uintp')
        else:
            cells = np.array([(i, (i+1)%nvertices) for i in range(nvertices)], dtype='uintp')

        return make_line_mesh(vertices, cells)
    
    ncells, nvtx_per_cell = cells.shape
    assert nvtx_per_cell == 2
    tdim = 1
    
    nvertices, gdim = vertices.shape
    assert gdim > 1

    mesh = df.Mesh()
    editor = df.MeshEditor()
    
    editor.open(mesh, 'interval', tdim, gdim)            

    editor.init_vertices(nvertices)
    editor.init_cells(ncells)

    for vi, x in enumerate(vertices): editor.add_vertex(vi, x)

    for ci, c in enumerate(np.asarray(cells, dtype='uintp')): editor.add_cell(ci, c)
    
    editor.close()

    return mesh
