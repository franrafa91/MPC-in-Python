import numpy as np
def spring(t, w, f= lambda t,w: 0, k=1E5, c=2E3, m=450):
    ''' Linear model of spring-mass-damper system

    Args:
    t: time (s)
    w: state vector [displacement (m), velocity (m/s)]
    f: external force function (N)
    k: spring constant (N/m)
    c: damping coefficient (N·s/m)
    m: mass (kg)
    '''

    # unpack state vector
    assert(np.array(w).size == 2)
    x, v, = w

    # compute derivatives
    dv = (-k*x - c*v + f(t,w))/m # Acceleration
    dx = v # Velocity

    return np.array([dx, dv])