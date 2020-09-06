from mbed.curve_distance import curve_distance
import dolfin as df
import numpy as np


def distance(A, B, P):
        '''Segment |AB| distance to P'''
        t = np.dot(B-A, P-A)/np.dot(B-A, B-A)

        d = np.sqrt(np.dot(P-A, P-A) + 2*t*np.dot(P-A, A-B) + t**2*np.dot(B-A, B-A))
        
        return min((np.linalg.norm(P - A),
                    np.linalg.norm(P - B),
                    d if 1E-13 < t < 1+1E-13 else np.inf))

    
def test_one_line():
    mesh = df.UnitCubeMesh(8, 8, 8)
    f = df.MeshFunction('size_t', mesh, 1, 0)
    df.CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2])').mark(f, 1)
    
    d = curve_distance(f, nlayers=10, outside_val=-1)
    # See if we got the distance right
    A, B = np.array([0., 0, 0]), np.array([1, 1., 1])

    dofs_x = d.function_space().tabulate_dof_coordinates().reshape((-1, 3))
    for xi in dofs_x:
        assert abs(d(xi) - distance(A, B, xi)) < 1E-13

        
def test_two_line():
    mesh = df.UnitCubeMesh(8, 8, 8)
    f = df.MeshFunction('size_t', mesh, 1, 0)
    df.CompiledSubDomain('near(x[0], x[1]) && near(x[1], x[2])').mark(f, 1)
        
    # One more line
    df.CompiledSubDomain('near(x[0], x[1]) && near(1., x[2])').mark(f, 1)
    
    d = curve_distance(f, nlayers=10, outside_val=-1)
    # See if we got the distance right
    A, B, C = np.array([0., 0, 0]), np.array([1, 1., 1]), np.array([0., 0, 1.])

    dofs_x = d.function_space().tabulate_dof_coordinates().reshape((-1, 3))
    for xi in dofs_x:
        assert abs(d(xi) - min((distance(A, B, xi), distance(C, B, xi)))) < 1E-13
