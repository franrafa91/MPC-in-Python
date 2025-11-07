import math
import numpy as np

def train_and_rail(t, w, f=lambda t,w: 0, m=100, g=9.81, theta=math.pi/6, mu=1e-3):
    def sign(x):
        return 0 if x==0 else (1 if x>0 else -1)

    x_box, y_box, vy_box, x_train, vx_train, y_train = w #unpack state vector

    free_fall = not((y_box-y_train)<=1 and abs(x_box-x_train)<1)

    dx_box = 0
    dy_box = vy_box if free_fall else 0
    dvy_box = -g if free_fall else 0
    dx_train = vx_train
    dvx_train = -g*math.sin(theta) - mu*g*math.cos(theta)*sign(vx_train) + math.cos(theta)*f(t,w)
    dy_train = vx_train/math.cos(theta)*math.sin(theta)

    return np.array([dx_box, dy_box, dvy_box, dx_train, dvx_train, dy_train])

def plot_animation(t_sol, w_sol, controller, SKIPFRAMES=2, theta=math.pi/6):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    rows = 4
    cols = 3
    x_max = 120
    t_span = [min(t_sol),max(t_sol)]
    fig = make_subplots(rows=rows, cols=cols,
                        specs=[[{"rowspan":3, "colspan":2}, None, {}],
                            [None, None, {}],
                            [None, None, {}],
                            [{}, {}, {}]],
                        subplot_titles=("PID Train Simulation", "x [m]", "v [m/s]", "a [m/s²]", "ε(x)", "Δε(x)", "∫Δε(x)"))

    x_box_sol, y_box_sol, vy_box_sol, x_train_sol, vx_train_sol, y_train_sol = w_sol
    a_sol = [0] + np.diff(vx_train_sol/t_sol[1],1).tolist()
    # Precompute frames
    len_t = len(t_sol)
    # Only create frames for sampled timesteps to speed up animation
    sampled_idx = list(range(0, len_t,SKIPFRAMES))
    frame_amount = len(sampled_idx)
    frames = []
    frame_meta = []
    for num in range(frame_amount):
        idx_t = sampled_idx[num]

        platform_x = [x_train_sol[idx_t]-3.1, x_train_sol[idx_t]+3.1]
        platform_y = [y_train_sol[idx_t], y_train_sol[idx_t]]
        cube_x = [x_box_sol[idx_t]-1, x_box_sol[idx_t]+1]
        cube_y = [y_box_sol[idx_t], y_box_sol[idx_t]]

        # traces for main (row1 col1)
        main_traces = [
            go.Scatter(x=[0, x_max], y=[5, x_max*np.tan(theta)+5], mode='lines', line=dict(color='black', width=6)),
            go.Scatter(x=platform_x, y=platform_y, mode='lines', line=dict(color='blue', width=18)),
            go.Scatter(x=cube_x, y=cube_y, mode='lines', line=dict(color='black', width=14)),
        ]

        # Add a time annotation (as a text trace) positioned in the main subplot
        time_text = go.Scatter(x=[x_max*0.05], y=[x_max*0.9],
                            mode='text', text=[f"t={t_sol[idx_t]:.2f}s"], textfont=dict(size=16, color='black'), textposition="bottom right")

        # time series traces (use partial data up to idx_t)
        ts_traces = [
            go.Scatter(x=t_sol[:idx_t+1], y=x_train_sol[:idx_t+1], mode='lines', line=dict(color='blue'), name='x'),
            go.Scatter(x=t_sol[:idx_t+1], y=vx_train_sol[:idx_t+1], mode='lines', line=dict(color='blue'), name='v'),
            go.Scatter(x=t_sol[:idx_t+1], y=a_sol[:idx_t+1], mode='lines', line=dict(color='blue'), name='a'),
            go.Scatter(x=t_sol[:idx_t+1], y=controller.e[:idx_t+1], mode='lines', line=dict(color='blue'), name='ε(x)'),
            go.Scatter(x=t_sol[:idx_t+1], y=controller.e_dot[:idx_t+1], mode='lines', line=dict(color='blue'), name='Δε(x)'),
            go.Scatter(x=t_sol[:idx_t+1], y=controller.e_int[:idx_t+1], mode='lines', line=dict(color='blue'), name='∫Δε(x)'),
        ]

        # Build frame with updates for each subplot index
        frame = go.Frame(data=main_traces + [time_text] + ts_traces, name=str(num))
        frames.append(frame)

    # Initial static traces (frame 0)
    init_main = frames[0].data[:4]  # now includes time_text and trial_text as 4th  in main group
    init_ts = frames[0].data[4:]

    # Place main traces into (1,1) (spans rows 1-3 cols 1-2)
    for tr in init_main:
        fig.add_trace(tr, row=1, col=1)

    # Place time series traces into their subplots
    fig.add_trace(init_ts[0], row=1, col=3)  # displ
    fig.add_trace(init_ts[1], row=2, col=3)  # vel
    fig.add_trace(init_ts[2], row=3, col=3)  # accel
    fig.add_trace(init_ts[3], row=4, col=1)  # error
    fig.add_trace(init_ts[4], row=4, col=2)  # e_dot
    fig.add_trace(init_ts[5], row=4, col=3)  # e_int

    # Layout adjustments: fix axis ranges so they don't autoscale per frame
    fig.update_xaxes(range=[0, x_max], row=1, col=1)
    fig.update_yaxes(range=[0, x_max], row=1, col=1)

    # Time-series axes ranges (fix across all relevant subplots)
    fig.update_xaxes(range=t_span, row=1, col=3)
    fig.update_xaxes(range=t_span, row=2, col=3)
    fig.update_xaxes(range=t_span, row=3, col=3)
    fig.update_xaxes(range=t_span, row=4, col=1)
    fig.update_xaxes(range=t_span, row=4, col=2)
    fig.update_xaxes(range=t_span, row=4, col=3)

    # Fix y-limits for time series to the min/max computed once so they don't change per frame
    def fixed_ylim(arr):
        vmin = np.min(arr)
        vmax = np.max(arr)
        padding = max(abs(vmin), abs(vmax)) * 0.1 if max(abs(vmin), abs(vmax))>0 else 1
        return vmin - padding, vmax + padding

    displ_ylim = fixed_ylim(x_train_sol)
    vrail_ylim = fixed_ylim(vx_train_sol)
    a_rail_ylim = fixed_ylim(a_sol)
    e_ylim = fixed_ylim(controller.e)
    edot_ylim = fixed_ylim(controller.e_dot)
    eint_ylim = fixed_ylim(controller.e_int)

    fig.update_yaxes(range=displ_ylim, row=1, col=3)
    fig.update_yaxes(range=vrail_ylim, row=2, col=3)
    fig.update_yaxes(range=a_rail_ylim, row=3, col=3)
    fig.update_yaxes(range=e_ylim, row=4, col=1)
    fig.update_yaxes(range=edot_ylim, row=4, col=2)
    fig.update_yaxes(range=eint_ylim, row=4, col=3)

    fig.frames = frames
    fig.update_layout(showlegend=False, height=600, width=1000,
                    title_text=None)

    # Animation settings — use FRAME_DURATION_MS and avoid full redraw when possible
    fig.update_layout(updatemenus=[dict(type='buttons', showactive=True,
                        y=1.05, x=1.10,
                        xanchor='right', yanchor='top',
                        buttons=[
                {"args": [None, {"frame": {"duration": 0, "redraw": False},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0,"easing": "quadratic-in-out"}}],
                                    "label": "Play/Pause","method": "animate",
                "args2": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                                    "label": "Play/Pause", "method": "animate"}
                                ]
                            )])

    # Disable global layout transition easing to avoid slow implicit transitions when frames change
    fig.update_layout(transition={"duration": 1, "easing": "linear"})

    # Build an explicit slider with immediate-mode steps and zero-duration transitions so
    # clicking/dragging the slider updates frames instantly (no implicit tweening).
    steps = []
    for num in range(len(frames)):
        idx_t = sampled_idx[num]        
        # show label only when time is exactly 0 (trial start) or a multiple of 0.5s
        label = f"{t_sol[idx_t]:.2f}s"
        step = dict(
            value=num,
            method='animate',
            label=label,
            visible=True,
            args=[[frames[num].name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
        )
        steps.append(step)

    # Use a fixed small top padding so the slider doesn't push the plotting area down when many steps exist
    slider = dict(
        active=0,
        currentvalue={"font": {"size": 14, "weight": 20, "color": "black"}},
        pad={"t": 0},
        steps=steps,
        x=0,
        xanchor='left',
        y=-0.05,
        yanchor='top',
        ticklen=0,
        minorticklen=0,
        font=dict(size=1,weight=1, color='white')
        )

    fig.update_layout(sliders=[slider])
    return fig