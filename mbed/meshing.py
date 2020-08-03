import mbed.point_embedding as point
import mbed.line_embedding as line
from mbed.utils import *
from mbed.bounding import *

from collections import namedtuple
import numpy as np
import gmsh


def embed_mesh1d(mesh1d, bounding_shape, how, *gmsh_args, **kwargs):
    '''
    Embed 1d in Xd mesh Xd padded hypercube. Embedding can be as 
    points / lines (these are gmsh options). 

    Returns: LineMeshEmbedding
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

    if 'debug' not in kwargs:
        kwargs['debug'] = False
        
    if 'save_embedding' not in kwargs:
        kwargs['save_embedding'] = ''
        kwargs['save_geo'] = ''
    else:
        d = kwargs['save_embedding']
        d and (not os.path.exists(d) and (os.makedirs(d)))
        
    if 'save_geo' in kwargs:
        if kwargs['save_embedding']:
            kwargs['save_geo'] = os.path.join(kwargs['save_embedding'], kwargs['save_geo'])
        else:
            kwargs['save_geo'] = ''

    print_blue('Emebdding %d vertices and %d edges in R^%d' % (mesh1d.num_vertices(),
                                                               mesh1d.num_cells(),
                                                               mesh1d.geometry().dim()))
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
