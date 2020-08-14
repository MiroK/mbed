from perfusion.boundary_conditions import poisson_solve, EdgeDirichletBC
import dolfin as df
import os


# NOTE: this is important to get the edges synced correctly
df.parameters['ghost_mode'] = 'shared_vertex'

embedding_folder = './rat_timo'

mesh = df.Mesh()
h5_file = df.HDF5File(mesh.mpi_comm(), .os.path.join(embedding_folder, 'mesh.h5'), 'r')
h5_file.read(mesh, 'embedding_mesh', False)


edge_f = df.MeshFunction('size_t', mesh, 1, 0)
h5_file.read(edge_f, 'edge_coloring')
h5_file.close()
# Color embedded segments
edge_f.array()[edge_f.array() > 0] = 1


V = df.FunctionSpace(mesh, 'CG', 1)
f = df.Constant(1)
bcs = EdgeDirichletBC(V, 2, edge_f, 1)

_, uh = poisson_solve(V, f, bcs)

# This better be very close if multiple CPUs are involved
print(uh.vector().norm('l2'))
