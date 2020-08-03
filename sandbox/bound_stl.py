from mbed.utils import hypercube
import numpy as np
import gmsh

# To get the bounding shape as stl
if False:
    x0, x1 = np.zeros(3), np.ones(3)
    padding = 0.1*np.ones(3)

    gmsh.initialize([])
    
    model = gmsh.model
    hypercube(model, x0, x1, padding)
    model.geo.synchronize()
    
    gmsh.fltk.initialize()
    gmsh.fltk.run()

    
# gmsh.initialize()
# gmsh.option.setNumber('General.Terminal', 1)
# gmsh.merge('box.stl')

# s = gmsh.model.getEntities(2)
# l = gmsh.model.geo.addSurfaceLoop([s[i][1] for i in range(len(s))])
# vol = gmsh.model.geo.addVolume([l])

# gmsh.model.geo.synchronize()

# a = gmsh.model.geo.addPoint(*[0.2, 0.2, 0.2])
# b = gmsh.model.geo.addPoint(*[0.4, 0.4, 0.4])
# l = gmsh.model.geo.addLine(a, b)

# gmsh.model.geo.synchronize()

# gmsh.model.mesh.embed(1, [l], 3, vol)

# gmsh.model.geo.synchronize()

# gmsh.fltk.initialize()
# gmsh.fltk.run()
