class ExplicitSolver:
    """
    Abstract class for Dynamical System Solvers
    """

    def __init__(self):
        raise NotImplementedError("Current Solver does not have this method implemented.")
    
    def step(self, df, t, w, dt):
        raise NotImplementedError("Current Solver does not have this method implemented.")

class FE(ExplicitSolver):
    def __init__(self):
        pass

    def step(self, df, t, w, dt):
        dw = df(t,w)
        return dw

class RK4(ExplicitSolver):
    def __init__(self):
        pass

    def step(self, df, t, w, dt):
        k1 = df(t, w)
        k2 = df(t + 0.5 * dt, w + 0.5 * dt * k1)
        k3 = df(t + 0.5 * dt, w + 0.5 * dt * k2)
        k4 = df(t + dt, w + dt * k3)
        dw = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return dw