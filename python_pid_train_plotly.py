"""
Plotly version of python_pid_train.py animation.
This reproduces the same simulation and creates an interactive Plotly figure with subplots
and an animated main scene. It saves an HTML file that can be opened in a browser.
"""
##%
import numpy as np
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Simulation parameters (kept same as original)
trials = 4
incl_angle = np.pi/6*1
g = 9.81
mass_cart = 100

K_p = 300
K_d = 250
K_i = 10

t0 = 0
dt = 0.02
t0 = 0
t_end = 5.0
t = np.arange(t0, t_end+dt, dt)

# Performance tuning: skip frames to reduce total frames and set animation speed (ms per frame)
# Increase FRAME_SKIP to make the animation coarser but much faster. Default 1 = every timestep.
FRAME_SKIP = 2
# Frame duration in milliseconds for Plotly animation
FRAME_DURATION_MS = 0

F_g = -mass_cart*g

def set_x_ref(incl_angle):
    rand_h = random.uniform(0,120)
    rand_v = random.uniform(20+120*np.tan(incl_angle)+6.5,40+120*np.tan(incl_angle)+6.5)
    return rand_h, rand_v

displ_rail = np.zeros((trials, len(t)))
v_rail = np.zeros((trials, len(t)))
a_rail = np.zeros((trials, len(t)))
pos_x_train = np.zeros((trials, len(t)))
pos_y_train = np.zeros((trials, len(t)))
e = np.zeros((trials, len(t)))
e_dot = np.zeros((trials, len(t)))
e_int = np.zeros((trials, len(t)))

pos_x_cube = np.zeros((trials, len(t)))
pos_y_cube = np.zeros((trials, len(t)))

F_ga_t = F_g*np.sin(incl_angle)
init_pos_x = 120
init_pos_y = 120*np.tan(incl_angle)+6.5
init_displ_rail = (init_pos_x**2+init_pos_y**2)**0.5
init_vel_rail = 0
init_a_rail = 0

init_pos_x_global = init_pos_x

trials_global = trials
trials_magn = trials
history = np.ones(trials)
while(trials>0):
    pos_x_cube_ref = set_x_ref(incl_angle)[0]
    pos_y_cube_ref = set_x_ref(incl_angle)[1]
    times = trials_magn-trials
    pos_x_cube[times] = pos_x_cube_ref
    pos_y_cube[times] = pos_y_cube_ref - g/2*t**2
    win = False
    delta = 1

    for i in range(1, len(t)):
        if i==1:
            displ_rail[times][0]=init_displ_rail
            pos_x_train[times][0]=init_pos_x
            pos_y_train[times][0]=init_pos_y
            v_rail[times][0]=init_vel_rail
            a_rail[times][0]=init_a_rail

        e[times][i-1]=pos_x_cube_ref-pos_x_train[times][i-1]

        if i>1:
            e_dot[times][i-1]=(e[times][i-1]-e[times][i-2])/dt
            e_int[times][i-1]=e_int[times][i-2]+(e[times][i-2]+e[times][i-1])/2*dt
        if i==len(t)-1:
            e[times][-1]=e[times][-2]
            e_dot[times][-1]=e_dot[times][-2]
            e_int[times][-1]=e_int[times][-2]

        F_a=K_p*e[times][i-1]+K_d*e_dot[times][i-1]+K_i*e_int[times][i-1]
        F_net=F_a+F_ga_t
        a_rail[times][i]=F_net/mass_cart
        v_rail[times][i]=v_rail[times][i-1]+(a_rail[times][i-1]+a_rail[times][i])/2*dt
        displ_rail[times][i]=displ_rail[times][i-1]+(v_rail[times][i-1]+v_rail[times][i])/2*dt
        pos_x_train[times][i]=displ_rail[times][i]*np.cos(incl_angle)
        pos_y_train[times][i]=displ_rail[times][i]*np.sin(incl_angle)+6.5

        if (pos_x_train[times][i]-5<pos_x_cube[times][i]+3 and pos_x_train[times][i]+5>pos_x_cube[times][i]-3) or win==True:
            if (pos_y_train[times][i]+3<pos_y_cube[times][i]-2 and pos_y_train[times][i]+8>pos_y_cube[times][i]+2) or win==True:
                win=True
                if delta==1:
                    change=pos_x_train[times][i]-pos_x_cube[times][i]
                    delta=0
                pos_x_cube[times][i]=pos_x_train[times][i]-change
                pos_y_cube[times][i]=pos_y_train[times][i]+5

    init_displ_rail=displ_rail[times][-1]
    init_pos_x=pos_x_train[times][-1]+v_rail[times][-1]*np.cos(incl_angle)*dt
    init_pos_y=pos_y_train[times][-1]+v_rail[times][-1]*np.sin(incl_angle)*dt
    init_vel_rail=v_rail[times][-1]
    init_a_rail=a_rail[times][-1]
    history[times]=delta
    trials=trials-1

# Build Plotly figure with subplots
rows = 4
cols = 3
fig = make_subplots(rows=rows, cols=cols,
                    specs=[[{"rowspan":3, "colspan":2}, None, {}],
                           [None, None, {}],
                           [None, None, {}],
                           [{}, {}, {}]],
                    subplot_titles=("PID Train Simulation", "x [m]", "v [m/s]", "a [m/s²]", "ε(x)", "Δε(x)", "∫Δε(x)"))

# Precompute frames
len_t = len(t)
# Only create frames for sampled timesteps to speed up animation
sampled_idx = list(range(0, len_t, FRAME_SKIP))
frame_amount = len(sampled_idx)*trials_global
frames = []
frame_meta = []
for num in range(frame_amount):
    idx_trial = int(num/len(sampled_idx))
    idx_t = sampled_idx[num - idx_trial*len(sampled_idx)]

    platform_x = [pos_x_train[idx_trial][idx_t]-3.1, pos_x_train[idx_trial][idx_t]+3.1]
    platform_y = [pos_y_train[idx_trial][idx_t], pos_y_train[idx_trial][idx_t]]
    cube_x = [pos_x_cube[idx_trial][idx_t]-1, pos_x_cube[idx_trial][idx_t]+1]
    cube_y = [pos_y_cube[idx_trial][idx_t], pos_y_cube[idx_trial][idx_t]]

    # traces for main (row1 col1)
    main_traces = [
        go.Scatter(x=[0, init_pos_x_global], y=[5, init_pos_x_global*np.tan(incl_angle)+5], mode='lines', line=dict(color='black', width=6)),
        go.Scatter(x=platform_x, y=platform_y, mode='lines', line=dict(color='blue', width=18)),
        go.Scatter(x=cube_x, y=cube_y, mode='lines', line=dict(color='black', width=14)),
    ]

    # Add a time annotation (as a text trace) positioned in the main subplot
    time_text = go.Scatter(x=[init_pos_x_global*0.05], y=[init_pos_x_global*0.9],
                           mode='text', text=[f"t={t[idx_t]:.2f}s"], textfont=dict(size=16, color='black'), textposition="bottom right")

    # Add trial number text below the time_text
    trial_text = go.Scatter(x=[init_pos_x_global*0.05], y=[init_pos_x_global*0.85],
                            mode='text', text=[f"trial {idx_trial+1}"], textfont=dict(size=14, color='black'), textposition="bottom right")

    # time series traces (use partial data up to idx_t)
    ts_traces = [
        go.Scatter(x=t[:idx_t+1], y=displ_rail[idx_trial][:idx_t+1], mode='lines', line=dict(color='blue'), name='x'),
        go.Scatter(x=t[:idx_t+1], y=v_rail[idx_trial][:idx_t+1], mode='lines', line=dict(color='blue'), name='v'),
        go.Scatter(x=t[:idx_t+1], y=a_rail[idx_trial][:idx_t+1], mode='lines', line=dict(color='blue'), name='a'),
        go.Scatter(x=t[:idx_t+1], y=e[idx_trial][:idx_t+1], mode='lines', line=dict(color='blue'), name='ε(x)'),
        go.Scatter(x=t[:idx_t+1], y=e_dot[idx_trial][:idx_t+1], mode='lines', line=dict(color='blue'), name='Δε(x)'),
        go.Scatter(x=t[:idx_t+1], y=e_int[idx_trial][:idx_t+1], mode='lines', line=dict(color='blue'), name='∫Δε(x)'),
    ]

    # Build frame with updates for each subplot index
    frame = go.Frame(data=main_traces + [time_text, trial_text] + ts_traces, name=str(num))
    frames.append(frame)
    # record metadata for this frame so slider labels match the shown time and trial
    frame_meta.append((idx_trial, float(t[idx_t])))

# Initial static traces (frame 0)
init_main = frames[0].data[:5]  # now includes time_text and trial_text as 4th and 5th traces in main group
init_ts = frames[0].data[5:]

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
fig.update_xaxes(range=[0, init_pos_x_global], row=1, col=1)
fig.update_yaxes(range=[0, init_pos_x_global], row=1, col=1)

# Time-series axes ranges (fix across all relevant subplots)
fig.update_xaxes(range=[t0, t_end], row=1, col=3)
fig.update_xaxes(range=[t0, t_end], row=2, col=3)
fig.update_xaxes(range=[t0, t_end], row=3, col=3)
fig.update_xaxes(range=[t0, t_end], row=4, col=1)
fig.update_xaxes(range=[t0, t_end], row=4, col=2)
fig.update_xaxes(range=[t0, t_end], row=4, col=3)

# Fix y-limits for time series to the min/max computed once so they don't change per frame
def fixed_ylim(arr):
    vmin = np.min(arr)
    vmax = np.max(arr)
    padding = max(abs(vmin), abs(vmax)) * 0.1 if max(abs(vmin), abs(vmax))>0 else 1
    return vmin - padding, vmax + padding

displ_ylim = fixed_ylim(displ_rail)
vrail_ylim = fixed_ylim(v_rail)
a_rail_ylim = fixed_ylim(a_rail)
e_ylim = fixed_ylim(e)
edot_ylim = fixed_ylim(e_dot)
eint_ylim = fixed_ylim(e_int)

fig.update_yaxes(range=displ_ylim, row=1, col=3)
fig.update_yaxes(range=vrail_ylim, row=2, col=3)
fig.update_yaxes(range=a_rail_ylim, row=3, col=3)
fig.update_yaxes(range=e_ylim, row=4, col=1)
fig.update_yaxes(range=edot_ylim, row=4, col=2)
fig.update_yaxes(range=eint_ylim, row=4, col=3)

fig.frames = frames
fig.update_layout(showlegend=False, height=800, width=1400,
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
for i in range(len(frames)):
    trial_i, time_i = frame_meta[i]
    # show label only when time is exactly 0 (trial start) or a multiple of 0.5s
    label = f"Trial {trial_i+1} - {time_i:.2f}s"
    step = dict(
        value=i,
        method='animate',
        label=label,
        visible=True,
        args=[[frames[i].name], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}]
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

import os
import webbrowser

# Save to HTML with a tiny JS injection that "warms" the frames on load.
out_html = 'python_pid_train_plotly.html'
# Generate base HTML
html = fig.to_html(full_html=True, include_plotlyjs='cdn')

# JS snippet: on load, animate to the last frame and then immediately to frame 0
# using mode 'immediate' and zero durations to avoid any tweening on first Play.
frames_count = len(frames)
injection = f"""
<script>
document.addEventListener('DOMContentLoaded', function() {{
    var attempts = 0;
    function warm() {{
        var gd = document.querySelector('.plotly-graph-div');
        if (!gd) {{
            attempts++;
            if (attempts < 20) setTimeout(warm, 100);
            return;
        }}
        try {{
            var last = {frames_count} - 1;
            if (last < 0) return;
            // Jump to last frame immediately, then back to 0 immediately
            Plotly.animate(gd, [String(last)], {{frame: {{duration:0, redraw:false}}, mode: 'immediate', transition: {{duration:0}}}})
            .then(function() {{
                Plotly.animate(gd, ['0'], {{frame: {{duration:0, redraw:false}}, mode: 'immediate', transition: {{duration:0}}}});
            }});
        }} catch (e) {{ console.error(e); }}
    }}
    warm();
}});
</script>
"""

# Insert injection before closing </body>
if '</body>' in html:
    html = html.replace('</body>', injection + '</body>')

# Write and open
with open(out_html, 'w', encoding='utf-8') as f:
    f.write(html)
file_url = 'file://' + os.path.abspath(out_html).replace('\\', '/')
webbrowser.open(file_url)
print(f'Plot saved to {out_html} and opened in your default browser.')
