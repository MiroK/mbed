import mbed.point_embedding as point
import mbed.line_embedding as line
from mbed.utils import is_number
from mbed.bounding import *

from collections import namedtuple
import numpy as np
import gmsh


def embed_mesh1d(mesh1d, bounding_shape, how, *gmsh_args, **kwargs):
    '''
    Embed 1d in Xd mesh Xd padded hypercube. Embedding can be as 
    points / lines (these are gmsh options). 

    Returns: LineMeshEmbedding, [status]
    '''
    assert mesh1d.topology().dim() == 1
    assert mesh1d.geometry().dim() > 1

    # Some convenience API where we expand 
    if isinstance(bounding_shape, str):
        bounding_shape = STLShape(bounding_shape)
        return embed_mesh1d(mesh1d, bounding_shape, how, *gmsh_args, **kwargs)
        
    # int -> padded
    if is_number(bounding_shape):
        bounding_shape = (bounding_shape, )*mesh1d.geometry().dim()
        return embed_mesh1d(mesh1d, bounding_shape, how, *gmsh_args, **kwargs)
    
    # Padded to relative box
    if isinstance(bounding_shape, (tuple, list, np.ndarray)):
        bounding_shape = BoundingBox(bounding_shape)
        return embed_mesh1d(mesh1d, bounding_shape, how, *gmsh_args, **kwargs)

    # At this point the type of bounding shape must check out
    assert isinstance(bounding_shape, BoundingShape)
    # FIXME: we should have that all 1d points are <= bounding shape
    #        later ... compute intersects for points on surface
    #        skip all for now

    model = gmsh.model
    gmsh.initialize(list(gmsh_args))
    gmsh.option.setNumber("General.Terminal", 1)

    if bounding_shape.is_inside(mesh1d, mesh1d.coordinates()):
        if how.lower() == 'as_lines':
            out = line.line_embed_mesh1d(model, mesh1d, bounding_shape, **kwargs)
            gmsh.finalize()
            return out
        
        assert how.lower() == 'as_points'
        out = point.point_embed_mesh1d(model, mesh1d, bounding_shape, **kwargs)
        gmsh.finalize()
        return out

    # FIXME: Doing points on manifolds is more involved in gmsh
    # because the surface needs to be broken apart (manually?)
    raise NotImplementedError
