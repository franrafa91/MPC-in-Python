class pid:
    def __init__(self, K_p, K_d, K_i, error=lambda t, w:0):
        self.dt = None
        self.K_p = K_p
        self.K_d = K_d
        self.K_i = K_i

        self.f_a = []
        self.e = []
        self.e_dot = []
        self.e_int = []

        self.error = error
    
    def __call__(self, t, w):
        e = self.error(t,w)
        self.e.append(e)
        if len(self.e)==1:
            self.e_dot.append(0)
            self.e_int.append(0)
        else:
            self.e_dot.append((self.e[-1]-self.e[-2])/self.dt)
            self.e_int.append(self.e_int[-1]+(self.e[-2]+self.e[-1])/2*self.dt)
        action = self.K_p*self.e[-1]+self.K_d*self.e_dot[-1]+self.K_i*self.e_int[-1]
        self.f_a.append(action)
        return action