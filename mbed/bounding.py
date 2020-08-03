from mbed.utils import hypercube
import numpy as np


class BoundingShape(object):
    '''Parent defining the API'''
    def create_volume(self, model, x):
        '''Enclose x adding into gmsh.model'''
        raise NotImplementedError

    def is_inside(self, mesh1d, x, tol=1E-10):
        '''Is x stricly inside the bounding volume'''
        raise NotImplementedError

    def is_on_boundary(self, mesh1d, x, tol=1E-10):
        '''Is x on the surface of bounding volume'''
        raise NotImplementedError


class BoundingBox(BoundingShape):
    '''Cartesian axis padded box relative to min/max coords of mesh1d'''
    def __init__(self, padding):
        assert all(p > 0 for p in padding)
        self.pad = padding

    def create_volume(self, model, x):
        '''Bound it'''
        xmin, xmax = x.min(axis=0), x.max(axis=0)

        padding = self.pad*(xmax - xmin)

        xmin = xmin - 0.5*padding
        xmax = xmax + 0.5*padding
        
        return hypercube(model, xmin, xmax)
    
    def is_inside(self, mesh1d, x, tol=1E-10):
        '''Is x stricly inside the bounding volume'''
        # The one from coordinatates
        xmin, xmax = mesh1d.coordinates().min(axis=0), mesh1d.coordinates().max(axis=0)

        padding = self.pad*(xmax - xmin)

        xmin = xmin - 0.5*padding
        xmax = xmax + 0.5*padding

        if x.ndim == 1:
            x = np.array([x])

        xx_min, xx_max = x.min(axis=0), x.max(axis=0)

        return np.all(xmin < xx_min - tol) and np.all(xx_max + tol < xmax) 
