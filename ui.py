import configparser

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, callback, State
import plotly.graph_objects as go

from constant import *
from models import Hexapod

# dimension control
def make_slider(range, id, value, step=1, updatemode='drag'):
    return dcc.Slider(range[0], range[1], step,
                      value=value,
                      id=id, marks=None,
                      updatemode = updatemode,
                      tooltip={"placement": "right", "always_visible": True})

dim_ctl_ids = ['Front', 'Middle', 'Side', 'coxa', 'Femur', 'Tibia']
dim_ctl_labels = [html.Label(id)  for id in dim_ctl_ids]
values = DEFAULT_DIMSIONS + DEFAULT_LEG_LENGTH
dim_ctl_sliders = [make_slider([1, 20], id, v) for id, v in zip(dim_ctl_ids, values)]
dim_ctl_widgets = [dbc.Row(dbc.Col(html.Label(dcc.Markdown(f"**Dimension Setting**")), width=12), justify='center')]
for l, s in zip(dim_ctl_labels, dim_ctl_sliders):
    dim_ctl_widgets.append(dbc.Row([dbc.Col(l, width=2, align='start'),
                                    dbc.Col(s, width=10, align='start', className="g-0")],
                                   justify='center'))
dim_ctl_widgets.append(dbc.Row(dbc.Col(dbc.Button("Reset Dimension", outline=True, color="primary", className="me-1", id='reset-dim'), width=12), align='center',className="mt-3",))
dim_ctl_widgets.append(dbc.Row(dbc.Col(dbc.Button("Reset Poses", outline=True, color="primary", className="me-1", id='reset-pose'), width=12), align='center',className="mt-3"))
dim_ctl_widgets.append(dbc.Row(dbc.Col(dbc.Button("Reset 3D View", outline=True, color="primary", className="me-1", id='reset-view'), width=12), align='center',className="mt-3"))

# leg patterns
leg_ctl_ids = ['alpha', 'beta', 'gamma']
leg_labels = [r'$\alpha$ (coxa-zaxis)', r'$\beta$ (femur-xaxis)', r'$\gamma$ (tibia-xaxis)']
leg_ctl_labels = [html.Div(dcc.Markdown(id, mathjax=True))  for id in leg_labels]
leg_ctl_sliders = [make_slider([-180, 180], id, 0) for id in leg_ctl_ids]
widgets = [html.Label(dcc.Markdown("*Legs share the same pose*"))]
for l, s in zip(leg_ctl_labels, leg_ctl_sliders):
    widgets.append(l)
    widgets.append(s)

leg_ctl_widgets = dbc.Card(
                    dbc.CardBody(
                        widgets
                    ),
                    className="mt-3",
                )

# forward kinematics
fk_leg_labels = []
for v in LEG_ID_NAMES.values():
    fk_leg_labels.append(dbc.Col(html.Label(dcc.Markdown(f"{v}")), width=2, align='center'))

fk_sliders = []
fk_leg_ctl_ids = ['-fk-alpha', '-fk-beta', '-fk-gamma']
fk_leg_seg_labels = [r'$\alpha$', r'$\beta$', r'$\gamma$']
fk_slider_ids = []
for id, l in zip(fk_leg_ctl_ids, fk_leg_seg_labels):
    l_s = []
    for v in LEG_ID_NAMES.values():
        # l_s.append(dbc.Col(html.Div(dcc.Markdown(l, mathjax=True)), width=1))
        s = make_slider([-180, 180], v + id, 0)
        fk_slider_ids.append(v + id)
        l_s.append(dbc.Col([html.Label(dcc.Markdown(l, mathjax=True)), s], width=2, align='center'))
    fk_sliders.append(dbc.Row(l_s))

fk_widgets = [dbc.Row(fk_leg_labels, align='center')] + fk_sliders
fk_ctl_widgets = dbc.Card(
                    dbc.CardBody(
                        fk_widgets
                    ),
                    className="mt-3",
                )

# inverse kinematics
ik_t_slider = []
ik_r_slider = []
for axis in AXIS_INDEX.keys():
    ts = make_slider([-1, 1], 'IK-T'+axis, 0, step=0.01)
    ik_t_slider.append(dbc.Col([html.Label(dcc.Markdown('T'+axis, mathjax=True)), ts], width=4))
    rs = make_slider([-30, 30], 'IK-R'+axis, 0, step=0.1)
    ik_r_slider.append(dbc.Col([html.Label(dcc.Markdown('R'+axis, mathjax=True)), rs], width=4))

ik_stance_slider = [
    dbc.Col([html.Label(dcc.Markdown('Hip Stance', mathjax=True)), make_slider([-60, 60], 'IK-Hip Stance', 0, step=1)], width=6),
    dbc.Col([html.Label(dcc.Markdown('Leg Stance', mathjax=True)), make_slider([-60, 60], 'IK-Leg Stance', 0, step=1)], width=6)
]

ik_ctl_widgets = dbc.Card(
                    dbc.CardBody(
                        [dbc.Row(ik_t_slider), dbc.Row(ik_r_slider), dbc.Row(ik_stance_slider)]
                    ),
                    className="mt-3",
                )

# walking gait
gait_timer = dcc.Interval(id='walking-timer',
                interval=20, # in milliseconds
                n_intervals=0,
                max_intervals=0
        )
gait_play_bt = dbc.Button("Play", outline=True, color="primary", className="me-1", id='gait-play')
gait_pause_bt = dbc.Button("Pause", outline=True, color="primary", className="me-1", id='gait-pause')
gait_step_bt = dbc.Button(">>Step", outline=True, color="primary", className="me-1", id='gait-step')

gait_ck = dcc.Checklist(
    [   {
            "label": html.Div(['Tripod'], style={'color': 'LightGreen', 'font-size': 20}),
            "value": 'is_tripod',
        },
        {
            "label": html.Div(['Forward'], style={'color': 'Gold', 'font-size': 20}),
            "value": 'is_forward',
        },
        {
            "label": html.Div(['Rotate'], style={'color': 'MediumTurqoise', 'font-size': 20}),
            "value": 'is_rotate',
        },
    ],
    value=['is_tripod', 'is_forward'],
    labelStyle={"display": "flex", "align-items": "center"},
    id='gait-ck'
)

lift_swing_slider = make_slider([10,40], id='LiftSwing', value=20, step=1, updatemode='mouseup')
hip_swing_slider = make_slider([10,40], id='HipSwing', value=12, step=1, updatemode='mouseup')
step_swing_slider = make_slider([5,20], id='GaitStep', value=10, step=1, updatemode='mouseup')
gait_speed_slider = make_slider([5,20], id='GaitSpeed', value=10, step=1, updatemode='mouseup')
gait_sliders = [lift_swing_slider, hip_swing_slider, step_swing_slider, gait_speed_slider]
gait_slider_label = []
for s in gait_sliders:
    gait_slider_label.append(html.Label(s.id))
gait_widget = dbc.Card(
                    dbc.CardBody(
                        dbc.Row([
                            dbc.Col(gait_slider_label, width=1),
                            dbc.Col(gait_sliders, width=4),
                            dbc.Col(gait_ck, width=3),
                            dbc.Col([gait_play_bt, gait_pause_bt, gait_step_bt], width=3),
                            gait_timer
                        ])
                    ),
                    className="mt-3",
                )

# configures
conf = configparser.ConfigParser()
conf.read('style.ini', encoding='utf-8')

def draw_robot(robot:Hexapod):
    # generate data for hexapod plotting
    body_mesh = go.Mesh3d(
            x=[p.x for p in robot.body.vertices],
            y=[p.y for p in robot.body.vertices],
            z=[p.z for p in robot.body.vertices],
            color=conf["robot plotter"]['body_color'],
            name='robot-body-mesh',
            showlegend=False,
            opacity=0.7,
            i=[0,1,0,0],
            j=[1,2,3,4],
            k=[3,3,4,5],
        )

    body_outline = go.Scatter3d(
        x=[p.x for p in robot.body.vertices] + [robot.body.vertices[0].x],
        y=[p.y for p in robot.body.vertices] + [robot.body.vertices[0].y],
        z=[p.z for p in robot.body.vertices] + [robot.body.vertices[0].z],
        name='robot-body_outline',
        marker=dict(color=conf["robot plotter"]['leg_color'],
                    size=int(conf["robot plotter"]['joint_size'])),
        line=dict(width=int(conf["robot plotter"]['body_outline_width'])),
        showlegend=False
    )

    head = go.Scatter3d(
        x=[robot.body.head.x],
        y=[robot.body.head.y],
        z=[robot.body.head.z],
        name='robot-head',
        marker=dict(color=conf["robot plotter"]['head_color'],
                    size=int(conf["robot plotter"]['head_size'])),
    )


    graph_data = [body_mesh, body_outline, head]

    for i in range(6):
        leg = go.Scatter3d(
            x=[p.x for p in robot.legs[i].points_global],
            y=[p.y for p in robot.legs[i].points_global],
            z=[p.z for p in robot.legs[i].points_global],
            name='robot-leg-' + str(i),
            marker=dict(color=conf["robot plotter"]['leg_color'],
                        size=int(conf["robot plotter"]['joint_size'])),
            line=dict(width=int(conf["robot plotter"]['leg_width'])),
            showlegend=False
        )
        graph_data.append(leg)
    
    support_mesh = go.Mesh3d(
            x=[p.x for p in robot.ground_contact_points.values()],
            y=[p.y for p in robot.ground_contact_points.values()],
            z=[p.z-0.01 for p in robot.ground_contact_points.values()],
            color=conf["robot plotter"]['body_color'],
            name='support-mesh',
            showlegend=False,
            opacity=0.2,
    )
    graph_data.append(support_mesh)

    robot_axis = [
        go.Scatter3d(
            x=[robot.body.cog.x],
            y=[robot.body.cog.y],
            z=[robot.body.cog.z],
            name='robot-axis',
            marker=dict(color=conf["axis"]['r_origin_color'],
                        size=int(conf["axis"]['origin_size']))
            ),
        # x-axis
        go.Scatter3d(
            x = [robot.body.cog.x, robot.x_axis.x],
            y = [robot.body.cog.y, robot.x_axis.y],
            z = [robot.body.cog.z, robot.x_axis.z],
            mode='lines',
            line=dict(color=conf["axis"]['color_x']),
            showlegend=False
        ),
        go.Scatter3d(
            x = [robot.body.cog.x, robot.y_axis.x],
            y = [robot.body.cog.y, robot.y_axis.y],
            z = [robot.body.cog.z, robot.y_axis.z],
            mode='lines',
            line=dict(color=conf["axis"]['color_y']),
            showlegend=False
        ),
        go.Scatter3d(
            x = [robot.body.cog.x, robot.z_axis.x],
            y = [robot.body.cog.y, robot.z_axis.y],
            z = [robot.body.cog.z, robot.z_axis.z],
            mode='lines',
            line=dict(color=conf["axis"]['color_z']),
            showlegend=False
        )
    ]
    
    world_axis = [
        go.Scatter3d(
            x=[0],
            y=[0],
            z=[0],
            name='world-axis',
            marker=dict(color=conf["axis"]['w_origin_color'],
                        size=int(conf["axis"]['origin_size']))
            ),
        # x-axis
        go.Scatter3d(
            x = [0, conf["axis"]['axis_size']],
            y = [0, 0],
            z = [0, 0],
            mode='lines',
            line=dict(color=conf["axis"]['color_x']),
            showlegend=False
        ),
        # y-axis
        go.Scatter3d(
            x = [0, 0],
            y = [0, conf["axis"]['axis_size']],
            z = [0, 0],
            mode='lines',
            line=dict(color=conf["axis"]['color_y']),
            showlegend=False
        ),
        # z-axis
        go.Scatter3d(
            x = [0, 0],
            y = [0, 0],
            z = [0, conf["axis"]['axis_size']],
            mode='lines',
            line=dict(color=conf["axis"]['color_z']),
            showlegend=False
        )
    ]
    graph_data += robot_axis + world_axis
    
    # ground
    s = int(conf['ground']['size'])
    ground_mesh = go.Mesh3d(
            x=[s/2, -s/2, -s/2, s/2],
            y=[s/2, s/2, -s/2, -s/2],
            z=[0, 0, 0, 0],
            color=conf['ground']['color'],
            name='ground',
            showlegend=False,
            opacity=0.2,
            showscale=False
            
        )
    
    graph_data += [ground_mesh]
    return graph_data


def update_robot_graph(fig, robot:Hexapod):
    # body mesh
    fig["data"][0]['x'] = [p.x for p in robot.body.vertices]
    fig["data"][0]['y'] = [p.y for p in robot.body.vertices]
    fig["data"][0]['z'] = [p.z for p in robot.body.vertices]
    
    # body outline
    fig["data"][1]['x'] = [p.x for p in robot.body.vertices] + [robot.body.vertices[0].x]
    fig["data"][1]['y'] = [p.y for p in robot.body.vertices] + [robot.body.vertices[0].y]
    fig["data"][1]['z'] = [p.z for p in robot.body.vertices] + [robot.body.vertices[0].z]

    # head
    fig["data"][2]['x'] = [robot.body.head.x]
    fig["data"][2]['y'] = [robot.body.head.y]
    fig["data"][2]['z'] = [robot.body.head.z]

    # robot legs
    for i in range(6):
        fig["data"][i+3]['x'] = [p.x for p in robot.legs[i].points_global]
        fig["data"][i+3]['y'] = [p.y for p in robot.legs[i].points_global]
        fig["data"][i+3]['z'] = [p.z for p in robot.legs[i].points_global]
    
    # support mesh
    fig["data"][9]['x'] = [p.x for p in robot.ground_contact_points.values()]
    fig["data"][9]['y'] = [p.y for p in robot.ground_contact_points.values()]
    fig["data"][9]['z'] = [p.z-0.01 for p in robot.ground_contact_points.values()]
    
    # robot axis
    fig["data"][10]['x'] = [robot.body.cog.x]
    fig["data"][10]['y'] = [robot.body.cog.y]
    fig["data"][10]['z'] = [robot.body.cog.z]
    
    fig["data"][11]['x'] = [robot.body.cog.x, robot.x_axis.x]
    fig["data"][11]['y'] = [robot.body.cog.y, robot.x_axis.y]
    fig["data"][11]['z'] = [robot.body.cog.z, robot.x_axis.z]
    
    fig["data"][12]['x'] = [robot.body.cog.x, robot.y_axis.x]
    fig["data"][12]['y'] = [robot.body.cog.y, robot.y_axis.y]
    fig["data"][12]['z'] = [robot.body.cog.z, robot.y_axis.z]
    
    fig["data"][13]['x'] = [robot.body.cog.x, robot.z_axis.x]
    fig["data"][13]['y'] = [robot.body.cog.y, robot.z_axis.y]
    fig["data"][13]['z'] = [robot.body.cog.z, robot.z_axis.z]
    
    return fig

def play_robot_walking(fig, robot:Hexapod, t):
    robot.set_pose_from_walking_sequence(t)
    return update_robot_graph(fig, robot)
    

# robot instance
robot = Hexapod()
graph_data = draw_robot(robot)

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=0, y=0.5, z=0.4)
)

fig = go.Figure(data=graph_data)
fig.update_layout(height=600,
                  uirevision=True,
                  scene={
                      "camera":camera,
                      "aspectmode": "manual",
                      "aspectratio": {"x": 1, "y": 1, "z": 1},
                      'xaxis':{"nticks":1, "backgroundcolor":"white", "range": [-20, 20], "tickfont":dict(color="white")},
                      'yaxis':{"nticks":1, "backgroundcolor":"white", "range": [-20, 20], "tickfont":dict(color="white")},
                      'zaxis':{"nticks":1, "backgroundcolor":"white", "range": [-20, 20], "tickfont":dict(color="white")},
                  })
# buttons
@callback(
    Output('Front', 'value'),
    Output('Middle', 'value'),
    Output('Side', 'value'),
    Output('coxa', 'value'),
    Output('Femur', 'value'),
    Output('Tibia', 'value'),
    Input('reset-dim', 'n_clicks'),
    prevent_initial_call=True
)
def reset_robot_dimension(btn):
    return DEFAULT_DIMSIONS + DEFAULT_LEG_LENGTH


@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('reset-view', 'n_clicks'),
    prevent_initial_call=True
)
def reset_camera_view(btn):
    fig['layout']['uirevision'] = False
    fig['layout']['scene']['camera'] = camera
    # fig['layout']['uirevision'] = True
    return fig


# robot control
# dimension
@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('Front', 'value'),
    Input('Middle', 'value'),
    Input('Side', 'value'),
    Input('coxa', 'value'),
    Input('Femur', 'value'),
    Input('Tibia', 'value'),
    prevent_initial_call=True
)
def change_robot_dimension(f, m, s, coxa, femur, tibia):
    if robot.update_dimensions([f, m, s, coxa, femur, tibia]):
        update_robot_graph(fig, robot)
    return fig
# leg patterns
@callback(
    Output('graph', 'figure'),
    Input('alpha', 'value'),
    Input('beta', 'value'),
    Input('gamma', 'value'),
    prevent_initial_call=True
)
def change_robot_leg_pattern(a,b,c):
    if robot.update_leg_pattern([a, b, c]):
        update_robot_graph(fig, robot)
    return fig

# forward kinematics
html_e = [Output('graph', 'figure', allow_duplicate=True)]
for eid in fk_slider_ids:
    html_e.append(Input(eid, "value"))
@callback(
    *html_e, prevent_initial_call=True
)
def forward_kinematics(*args):
    poses = {}
    for leg, leg_id in LEG_NAMES_ID.items():
        a = {}
        for seg, seg_id in LEG_SEG_NAMES_ID.items():
            a[seg] = args[int(leg_id + seg_id*6)]
        poses[leg] = a
    if robot.update_leg_pose(poses):
        update_robot_graph(fig, robot)
    return fig

# inverse kinematics
@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('IK-RX', 'value'),
    Input('IK-RY', 'value'),
    Input('IK-RZ', 'value'),
    Input('IK-TX', 'value'),
    Input('IK-TY', 'value'),
    Input('IK-TZ', 'value'),
    prevent_initial_call=True
)
def inverse_kinematics(rx, ry, rz, tx, ty, tz):
    tx *= robot.body.f
    ty *= robot.body.s
    tz *= robot.legs[0].lengths[-1]
    robot.solve_ik([rx, ry, rz], [tx, ty, tz])
    update_robot_graph(fig, robot)
    return fig

# gait
@callback(
    Output('graph', 'figure', allow_duplicate=True),
    Input('walking-timer', 'n_intervals'),
    prevent_initial_call=True
)
def walking(n):
    play_robot_walking(fig, robot, n%len(robot.walking_sequence[0]['coxa']))
    return fig

@callback(
    Output('walking-timer', 'max_intervals', allow_duplicate=True),
    Input('gait-play', 'n_clicks'),
    prevent_initial_call=True
)
def play_gait(n):
    return -1

@callback(
    Output('walking-timer', 'max_intervals'),
    Input('gait-pause', 'n_clicks'),
    prevent_initial_call=True
)
def pause_gait(n):
    return 0

@callback(
    Output('walking-timer', 'n_intervals'),
    Input('gait-step', 'n_clicks'),
    State('walking-timer', 'n_intervals'),
    prevent_initial_call=True
)
def pause_gait(s, n):
    return s + 1

@callback(
    Output('walking-timer', 'n_intervals', allow_duplicate=True),
    Input('LiftSwing', 'value'),
    Input('HipSwing', 'value'),
    Input('GaitStep', 'value'),
    Input('GaitSpeed', 'value'),
    Input('gait-ck', 'value'),
    prevent_initial_call=True
)
def update_gait_parameters(ls, hs, st, sp, ck):
    para = {}
    para['HipSwing'] = hs
    para['LiftSwing'] = ls
    para['StepNum'] = st
    para['Speed'] = sp
    para['Gait'] = 'Tripod' if 'is_tripod' in ck else 'Ripple'
    para['Direction'] = 1 if 'is_forward' in ck else -1
    para['Rotation'] = 1 if 'is_rotate' in ck else 0
    robot.generate_walking_sequence(para)
    print(para)
    return 0

if __name__ == "__main__":
    app = dash.Dash(
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    
    app.layout = html.Div(
        dbc.Container(
            [
                dbc.Row([
                    dbc.Col(dim_ctl_widgets, width={"size": 3, "offset": 0}, align='center'),
                    dbc.Col(
                        html.Div(dcc.Graph(figure=fig, id='graph')),
                        width={"size": 9, "offset": 0},
                    )
                ], align='center', justify='center'),
                
                dbc.Row([
                    dcc.Tabs(id="page-tabs", value='leg-patterns', 
                            children=[
                                dcc.Tab(children=leg_ctl_widgets, label='Leg Pattern', value='leg-patterns'),
                                dcc.Tab(children=fk_ctl_widgets,label='Forward Kinematics', value='FK'),
                                dcc.Tab(children=ik_ctl_widgets,label='Inverse Kinematics', value='IK'),
                                dcc.Tab(children=gait_widget, label='Walking Gaits', value='Walk'),
                            ])
                ]),
            ]
        )
    )
    
    app.run_server(debug=True)