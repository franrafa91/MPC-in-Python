import numpy as np
from solvers.explicit import Solver

class SystemInterface:
    """
    Base abstract class for systems, providing a skeleton for operations such as
    initializing, solving, and plotting animations.
    """

    def __init__(self):
        """
        Initialize the system with a given set of parameters.
        """
        raise NotImplementedError("This method is not implemented for this System.")

    def __call__(self, t, w, f=lambda t,w: 0, **params):
        """
        Provide the change in every state variable as the result of the current
        time, state space and inputs applied to the system.

        Args:
            t (float): Current time
            w (np.array[float]): Current composition of the state space
            f (function): Function that defines the forces acting as external input to the system

        Returns:
            np.array[float]: Derivative of the System
        """
        raise NotImplementedError("This method is not implemented for this System.")
    
    def get_state_description(self):
        """
        Get list of strings with the expected structure of the state space vector
        """
        return NotImplementedError("This method is not implemented for this System.")
    
    def solve(self, solver:Solver, w_0:np.array, t_span, n_steps=1000, controller=None, f_ext = lambda t,w:0):
        """
        Develop a Forward Euler Simulation of the System for a specific initial condition, timespan,
        number of steps, controller, and external force function.
        """
        ## Initialize Solution Array
        t_sol = np.linspace(*t_span, n_steps)
        w_sol_T = np.ndarray((t_sol.size,w_0.size))
        w_sol_T[0] = w_0

        ## Solve
        for i in range(t_sol.size-1):
            if controller: controller.dt = t_sol[1]-t_sol[0]
            contr_input = 0 if not controller else controller(t_sol[i],w_sol_T[i]) # Controller action as input TODO: Change to action vector
            u = lambda t,w: f_ext(t,w)+contr_input # Function must probably be a input vector function
            df = lambda t,w: self(t,w,u)
            dt = t_sol[i+1]-t_sol[i]
            dw = solver.step(df,t_sol[i],w_sol_T[i],dt)
            w_sol_T[i+1] = w_sol_T[i] + dw*dt
        w_sol = w_sol_T.T
        return t_sol, w_sol

    def solve_fe(self, w_0:np.array, t_span, n_steps=1000, controller=None, f_ext = lambda t,w:0):
        """
        Develop a Forward Euler Simulation of the System for a specific initial condition, timespan,
        number of steps, controller, and external force function.
        """
        ## Initialize Solution Array
        t_sol = np.linspace(*t_span, n_steps)
        w_sol_T = np.ndarray((t_sol.size,w_0.size))
        w_sol_T[0] = w_0

        ## Solve
        for i in range(t_sol.size-1):
            if controller: controller.dt = t_sol[1]-t_sol[0]
            contr_input = (lambda t,w: 0) if not controller else controller(t_sol[i],w_sol_T[i])
            f_tot = lambda t,w: f_ext(t,w)+contr_input
            dw = self(t_sol[i],w_sol_T[i],f_tot)
            dt = t_sol[i+1]-t_sol[i]
            w_sol_T[i+1] = w_sol_T[i] + dw*dt
        w_sol = w_sol_T.T
        return t_sol, w_sol

    def plot_sol(self, t_sol:np.array, w_sol:np.ndarray, labels:list=None, title:str=None, legend:str=None, plot:list=None):
        ## TODO: Should this really be a method of the system?
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        n_vars = w_sol.shape[0]

        if isinstance(plot,list):
            assert(len(plot)==n_vars)
        else:
            plot = [1 for i in range(n_vars)]
        plot = np.bool(plot)
        n_plots = int(sum(plot))

        # Treat Labels
        labels_def = [f'x{i+1}' for i in range(n_vars)]
        if isinstance(labels,list):
            labels = [labels[i] if len(labels)>i else labels_def[i] for i in range(n_vars)]
        else:
            labels = labels_def
        
        # Make Figure and Subplots
        fig = make_subplots(rows=n_plots, cols=1)
        legend = legend if legend else 'Sol 1'

        for i in range(n_vars):
            if plot[i]==1:
                plotnum = int(sum(plot[:i+1]))
                fig.add_trace(go.Scatter(x=t_sol, y=w_sol[i], name=legend, legendgroup=plotnum, legendgrouptitle_text=labels[i]), row=plotnum, col=1)
                fig.update_xaxes(dict(showticklabels=False), row=plotnum, col=1)
                fig.update_yaxes(dict(title_text=labels[i]), row=plotnum, col=1)
        fig.update_xaxes(dict(title_text="Time",showticklabels=True), row=n_plots, col=1)
        fig.update_layout(dict(title=title, legend_tracegroupgap = 200/n_plots))
        fig.update_layout(legend=dict(groupclick="toggleitem"))
        return fig

    def add_sol(self, fig, t_sol, w_sol, legend:str=None, plot:list=None):
        ## TODO: Should this really be a method of the system?
        import plotly.graph_objects as go

        n_vars = w_sol.shape[0]

        if isinstance(plot,list):
            assert(len(plot)==n_vars)
        else:
            plot = [1 for i in range(n_vars)]
        plot = np.bool(plot)
        n_plots = int(sum(plot))

        assert(n_plots == len(set([fig.to_dict()['data'][i]['legendgroup'] for i in range(len(fig.to_dict()['data']))])))
        n_sols = len(fig.to_dict()['data'])/n_plots
        legend = legend if legend else f'Sol {n_sols+1:g}'
        for i in range(n_vars):
            if plot[i]==1:
                plotnum = int(sum(plot[:i+1]))
                fig.add_trace(go.Scatter(x=t_sol, y=w_sol[i], name=legend, legendgroup=plotnum), row=plotnum, col=1)
        return fig