import mbed.point_embedding as point
import mbed.line_embedding as line
from collections import namedtuple
import numpy as np
import gmsh


def embed_mesh1d(mesh1d, padding, how, *gmsh_args, **kwargs):
    '''
    Embed 1d in Xd mesh Xd padded hypercube. Embedding can be as 
    points / lines (these are gmsh options). 

    Returns: LineMeshEmbedding, [status]
    '''
    assert mesh1d.topology().dim() == 1
    assert mesh1d.geometry().dim() > 1

    npoints, gdim = mesh1d.num_vertices(), mesh1d.geometry().dim()
    # Padding here is relative to dx
    if isinstance(padding, (int, float)):
        padding = padding*np.ones(gdim)
    else:
        padding = np.fromiter(padding, dtype=float)
    assert all(pad >= 0 for pad in padding)

    model = gmsh.model
    gmsh.initialize(list(gmsh_args))
    gmsh.option.setNumber("General.Terminal", 1)

    if all(pad > 0 for pad in padding):
        if how.lower() == 'as_lines':
            out = line.line_embed_mesh1d(model, mesh1d, padding, **kwargs)
            gmsh.finalize()
            return out
        
        assert how.lower() == 'as_points'
        out = point.point_embed_mesh1d(model, mesh1d, padding, **kwargs)
        gmsh.finalize()
        return out

    # FIXME: Doing points on manifolds is more involved in gmsh
    # because the surface needs to be broken apart (manually?)
    raise NotImplementedError
