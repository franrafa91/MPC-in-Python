import numpy as np

class ImplicitSolver:
    """
    Abstract class for Dynamical System Solvers
    """

    def __init__(self):
        raise NotImplementedError("Current Solver does not have this method implemented.")
    
    def step(self, df, t, w, dt):
        raise NotImplementedError("Current Solver does not have this method implemented.")
    
class radauIA(ImplicitSolver):
    def __init__(self, epsilon = 1e-6, tol=1e-6, maxit=10):
        self.epsilon = epsilon
        self.tol = tol
        self.maxit = maxit

    def step(self, df, t, w, dt):
        N = len(w)
        
        A = np.array([[1/4, -1/4],
                    [1/4, 5/12]])
        b = np.array([1/4, 3/4])
        c = np.array([0, 2/3])
        s = 2

        z = np.zeros(N * s)
        F = np.zeros(N * s)
        J = jac(self, df, t, w)

        for k in range(self.maxit):
            F[:N] = df(t[n] + c[0] * dt, w + z[:N])
            F[N:] = df(t[n] + c[1] * dt, w + z[N:])
            g = z - dt * np.dot(np.kron(A, np.eye(N)), F)
            delta_z = np.linalg.solve(np.eye(N * s) - dt * np.kron(A, J), -g)
            z += delta_z
            if np.linalg.norm(delta_z) < self.tol:
                break
        
        dw = np.dot(np.kron(b.T, np.eye(N)), F)
        return dw
    
def jac(solver:ImplicitSolver, df, t, w):
    J = np.zeros((len(w), len(w)))
    for i in range(len(w)):
        y_plus_epsilon = w.astype(float)
        y_minus_epsilon = w.astype(float)
        y_plus_epsilon[i] += solver.epsilon
        y_minus_epsilon[i] -= solver.epsilon
        J[:,i] = (df(t, y_plus_epsilon) - df(t,y_minus_epsilon)) / (2 * solver.epsilon)
    return J