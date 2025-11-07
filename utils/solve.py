import numpy as np
def solve(f, t_span, w_0:np.array, f_ext = lambda t,w:0, n_steps=1000,controller=None):
    ## Initialize Solution Array
    t_sol = np.linspace(*t_span, n_steps)
    w_sol_T = np.ndarray((t_sol.size,w_0.size))
    w_sol_T[0] = w_0

    ## Solve
    for i in range(t_sol.size-1):
        if controller: controller.dt = t_sol[1]-t_sol[0]
        f_cont = (lambda t,w: 0) if not controller else controller
        f_tot = lambda t,w: f_ext(t,w)+f_cont(t,w)
        dw = f(t_sol[i],w_sol_T[i],f_tot)
        dt = t_sol[i+1]-t_sol[i]
        w_sol_T[i+1] = w_sol_T[i] + dw*dt
    w_sol = w_sol_T.T
    return t_sol, w_sol

def plot_sol(t_sol:np.array, w_sol:np.ndarray, labels:list=None, title:str=None, legend:str=None, plot:list=None):
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

def add_sol(fig, t_sol, w_sol, legend:str=None, plot:list=None):
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