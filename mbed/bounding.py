from mbed.utils import hypercube
import numpy as np
import gmsh
import os


class BoundingShape(object):
    '''Parent defining the API'''
    def create_volume(self, model, x):
        '''Enclose x adding into gmsh.model'''
        raise NotImplementedError

    def is_inside(self, mesh1d, x, model=None, tol=1E-10):
        '''Is x stricly inside the bounding volume'''
        raise NotImplementedError

    def is_on_boundary(self, mesh1d, x, model=None, tol=1E-10):
        '''Is x on the surface of bounding volume'''
        raise NotImplementedError


class STLShape(BoundingShape):
    '''Bounding shape loaded from STL file'''
    def __init__(self, stl_path):
        _, ext = os.path.splitext(stl_path)
        assert ext == '.stl'
        self.stl_path = stl_path

    def is_inside(self, mesh1d, x, model=None, tol=1E-10):
        '''Is x stricly inside the bounding volume'''
        return True

    def create_volume(self, model, x):
        gmsh.merge(self.stl_path)

        s = gmsh.model.getEntities(2)
        l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
        vol = gmsh.model.geo.addVolume([l])

        gmsh.model.geo.synchronize()

        return {3: vol}


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
    
    def is_inside(self, mesh1d, x, model=None, tol=1E-10):
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
