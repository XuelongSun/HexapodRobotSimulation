import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation
from models import Body, Leg, Hexapod

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')

hexapod = Hexapod([3,2,3])

# draw body
head = ax.scatter(hexapod.body.head.x,
                  hexapod.body.head.y, 
                  hexapod.body.head.z,
                  facecolors='red', edgecolors='tomato', s=hexapod.body.f*80, zorder=1)
v = list([[v.x, v.y, v.z] for v in hexapod.body.vertices])
body = ax.add_collection3d(Poly3DCollection([v], facecolor='w', edgecolor='k', linewidth=8, zorder=0))
# draw legs
legs = {}
for k, leg in hexapod.legs.items():
    px = [leg.points_global[i].x for i in range(4)]
    py = [leg.points_global[i].y for i in range(4)]
    pz = [leg.points_global[i].z for i in range(4)]
    l, = ax.plot(px, py, pz, lw=8, color='k',
                 marker='o', markersize=10, mec='k', mfc='k')
    legs[k] = l

tip_curve, = ax.plot([], [], lw=1, color='r')
tip_curve_counter = 20
tip_leg = 5
tip_p = hexapod.legs[tip_leg].get_ground_contact_point()
tip_points = [[tip_p.x], [tip_p.y], [tip_p.z]]

t = np.linspace(0, np.pi*2, 100)
rot_x = -10*np.sin(t)
rot_y = -5*np.cos(t)
rot_z = -5*np.sin(t)
# ax.set_axis_off()
hexapod.generate_walking_sequence(dict(
    Gait='Tripod', HipSwing=30, LiftSwing=60, StepNum=10, Direction=-1, Rotation=0
))

def update(frame):
    global tip_points
    # IK
    hexapod.solve_ik([rot_x[frame%len(t)], rot_y[frame%len(t)], 0], [0, 0, 0])
    # Gait
    # hexapod.set_pose_from_walking_sequence(frame%len(hexapod.walking_sequence[0]['coxia']))
    # update body
    v = list([[v.x, v.y, v.z] for v in hexapod.body.vertices])
    body.set_verts([v])
    # update head
    head._offsets3d = ([hexapod.body.head.x], [hexapod.body.head.y], [hexapod.body.head.z])
    # update legs
    for k, leg in hexapod.legs.items():
        px = [leg.points_global[i].x for i in range(4)]
        py = [leg.points_global[i].y for i in range(4)]
        pz = [leg.points_global[i].z for i in range(4)]
        legs[k].set_data(px, py)
        legs[k].set_3d_properties(pz)
    # update tip curve
    tip_p = hexapod.legs[tip_leg].get_ground_contact_point()
    if frame == 0:
        tip_points = [[tip_p.x], [tip_p.y], [tip_p.z]]
    else:
        tip_points[0].append(tip_p.x)
        tip_points[1].append(tip_p.y)
        tip_points[2].append(tip_p.z)
    if len(tip_points[0]) < tip_curve_counter:
        tip_curve.set_data(tip_points[0], tip_points[1])
        tip_curve.set_3d_properties(tip_points[2])
    else:
        tip_curve.set_data(tip_points[0][-tip_curve_counter:-1],
                           tip_points[1][-tip_curve_counter:-1])
        tip_curve.set_3d_properties(tip_points[2][-tip_curve_counter:-1])
        
ax.set_aspect('equal')

# ani = FuncAnimation(fig, update, frames=4*len(hexapod.walking_sequence[0]['coxia']), interval=20, blit=False)
ani = FuncAnimation(fig, update, frames=4*len(t), interval=20, blit=False)
plt.show()