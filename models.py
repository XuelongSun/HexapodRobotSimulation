from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

from constant import *
from mathematics import *

class Body:
    #       |-f-|
    #      P2---*---P1--------
    #      /    |    \     |
    #     /     |     \    s
    #    /      |      \   |
    #   P3-------------P0 ---
    #    \      |      /|
    #     \     |     / |
    #      \    |    /  |
    #       P4--*--P5   |
    #           |       |
    #           |---m---|
    #   y axis
    #    ^
    #    |
    #    |
    #    ----> x axis
    def __init__(self, f=5, s=10, m=10) -> None:
        self.init_vertices((f,m,s))
    
    def init_vertices(self, dim):
        self.f, self.s, self.m = dim
        self.vertices = [
            Point3D(self.m, 0, 0, 'P0'),
            Point3D(self.f, self.s, 0, 'P1'),
            Point3D(-self.f, self.s, 0, 'P2'),
            Point3D(-self.m, 0, 0, 'P3'),
            Point3D(-self.f, -self.s, 0, 'P4'),
            Point3D(self.f, -self.s, 0, 'P5')
        ]
        self.cog = Point3D(0, 0, 0, 'COG')
        self.head = Point3D(0, self.s, 0, 'Head')
    
    def translate(self, offset):
        if hasattr(offset, "__len__"):
            if len(offset) == 3:
                for p in self.vertices + [self.cog, self.head]:
                    p.set_coordinates([p.x + offset[0], p.y + offset[1], p.z + offset[2]])
            else:
                raise ValueError
        else:
            raise ValueError

    def rotate(self, rot):
        if hasattr(rot, "__len__"):
            if len(rot) == 3:
                r = R.from_euler('XYZ', rot, degrees=True).as_matrix()
                for p in self.vertices + [self.cog, self.head]:
                    p.set_coordinates(np.matmul(r, p.get_coordinates()))
            else:
                raise ValueError
        else:
            raise ValueError
    
    def transform(self, transform):
        for v in self.vertices:
            v.set_coordinates(transform.dot(v.get_coordinates_homo()))
        self.head.set_coordinates(transform.dot(self.head.get_coordinates_homo()))
        self.cog.set_coordinates(transform.dot(self.cog.get_coordinates_homo()))
    
    def change_dimensions(self, dimension):
        self.init_vertices(dimension)
    
    def visualize2d(self, fig=None, ax=None):
        if fig is None:
            fig, ax = plt.subplots()
        # add head
        ax.scatter(self.head.x, self.head.y, facecolors='red', edgecolors='tomato', alpha=0.7, s=self.f*20)
        ax.text(self.head.x, self.head.y, 'Head')
        # add center of gravity
        ax.scatter(self.cog.x, self.cog.y, facecolors='k', edgecolors='gray', alpha=0.7, s=self.f*10)
        ax.text(self.cog.x, self.cog.y, 'COG')
        # add body hexagon
        v = [(v.x, v.y) for v in self.vertices]
        body = Polygon(v, facecolor='skyblue', alpha=0.6, fill=True, edgecolor='darkblue')
        ax.add_patch(body)
        # add point label
        for v in self.vertices:
            ax.text(v.x, v.y, v.name)
        
        # adjuestment
        ax.set_xlim([self.cog.x-1.5*self.m, self.cog.x + 1.5*self.m])
        ax.set_ylim([self.cog.y-1.5*self.s, self.cog.y + 1.5*self.s])
        ax.grid()
        ax.set_aspect('equalxy')
        return fig, ax
    
    def visualize3d(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure()
        ax = fig.add_subplot(projection="3d", proj_type="ortho")
        # add head
        ax.scatter(self.head.x, self.head.y, self.head.z, facecolors='red', edgecolors='tomato', alpha=0.7, s=self.f*20)
        ax.text(self.head.x, self.head.y, self.head.z, 'Head')
        # add center of gravity
        ax.scatter(self.cog.x, self.cog.y, self.cog.z, facecolors='k', edgecolors='gray', alpha=0.7, s=self.f*10)
        ax.text(self.cog.x, self.cog.y, self.cog.z, 'COG')
        v = list([[v.x, v.y, v.z] for v in self.vertices])
        ax.add_collection3d(Poly3DCollection([v], facecolor='skyblue', alpha=0.6, edgecolor='darkblue', linewidth=5))
        ax.set_xlim([self.cog.x-1.5*self.m, self.cog.x + 1.5*self.m])
        ax.set_ylim([self.cog.y-1.5*self.s, self.cog.y + 1.5*self.s])
        ax.set_zlim([-5, 5])
        ax.set_aspect('equal')
        return fig, ax


class Leg:
    #   |--lengths[0]-|--lengths[1]--|
    #   |=============|==============| p2 -------
    #   p0            p1             |          |
    # (origin)                       |          |
    #                                |  lengths[2]
    #                                |          |
    #                                |          |
    #                                | p3 -------
    #  z axis
    #  |
    #  |
    #  |------- y axis
    # origin
    def __init__(self, lengths=[10, 10, 10],
                 name='none', joint_num=3) -> None:
        self.joint_num = joint_num
        self.name=name
        
        self.lengths = lengths
        self.angles = [0, 0, 0]
        self.reset_pose()
        self.reset_transforms()
        self.global_transform = np.identity(4)
        # limitations of the angels / degrees
        self.angle_limits = [(-180, 180), (-180, 180), (-180, 180)]
        
    def __repr__(self):
        s = f"Leg:{self.name}({self.origin})"
        return s
    
    def reset_pose(self):
        self.angles = [0, 0, 0]
        p0 = Point3D(0, 0, 0, name='P0-BodyContact')
        p1 = Point3D(0, self.lengths[0], 0, name='P1-coxa')
        p2 = Point3D(0, self.lengths[0] + self.lengths[1], 0, name='P2-femur')
        p3 = Point3D(0, self.lengths[0] + self.lengths[1] + self.lengths[2], 0, name='P3-tibia')
        self.points_global = [p0, p1, p2, p3]
        
    def reset_transforms(self):
        # axes
        t_p0 = get_transformation_homo([0, 0, 0], [0, 0, 0])
        t_p1_p0 =  get_transformation_homo([0, 0, 0], [0, self.lengths[0], 0])
        # t_p1 = np.matmul(t_p0, t_p1_p0)
        t_p1 = t_p1_p0.dot(t_p0)
        t_p2_p1 =  get_transformation_homo([0, 0, 0], [0, self.lengths[1], 0])
        # t_p2 = np.matmul(t_p1, t_p2_p1)
        t_p2 = t_p2_p1.dot(t_p1)
        t_p3_p2 =  get_transformation_homo([0, 0, 0], [0, self.lengths[2], 0])
        # t_p3 = np.matmul(t_p2, t_p3_p2)
        t_p3 = t_p3_p2.dot(t_p2)
        self.transforms = [t_p0, t_p1, t_p2, t_p3]
        return self.transforms
    
    def _update_transforms(self):
        t_p0 = get_transformation_homo([0, 0, self.angles[0]], [0, 0, 0])
        t_p1_p0 =  get_transformation_homo([self.angles[1], 0, 0], [0, self.lengths[0], 0])
        t_p1 = t_p0.dot(t_p1_p0)
        t_p2_p1 =  get_transformation_homo([self.angles[2], 0, 0], [0, self.lengths[1], 0])
        t_p2 = t_p1.dot(t_p2_p1)
        t_p3_p2 =  get_transformation_homo([0, 0, 0], [0, self.lengths[2], 0])
        t_p3 = t_p2.dot(t_p3_p2)
        self.transforms = [t_p0, t_p1, t_p2, t_p3]
        
    def _update_pose(self):
        for p, t in zip(self.points_global, self.transforms):
            p.set_coordinates(self.global_transform.dot(t.dot([0, 0, 0, 1])))
            p.name = 'BodyCOG-' + p.name
    
    def change_pose(self, angles):
        self.angles = angles
        self._update_transforms()
        self._update_pose()
        return self.points_global
    
    def get_ground_contact_point(self):
        # the lowest point as the ground contact
        # usual case is the end point of the leg, i.e., tibia
        self.ground_contact_point = self.points_global[-1]
        for p in self.points_global[::-1]:
            if p.z < self.ground_contact_point.z:
                self.ground_contact_point = p
        return self.ground_contact_point
    
    def transform(self, transform):
        self.global_transform = np.matmul(transform, self.global_transform)
        # self.global_transform = transform
        for p in self.points_global:
            p.set_coordinates(transform.dot(p.get_coordinates_homo()))
    
    def solve_ik(self, start_p, end_p):
        vec_p0_p3 = end_p - start_p
        vec_p0_p3_len = np.linalg.norm(vec_p0_p3)
        # coxa vector is the projection of the P0->P3 onto the xy-plane
        coxa_vec = project_vector_onto_plane(vec_p0_p3, np.array((0, 0, 1)))
        coxa_vec_unit = coxa_vec/np.linalg.norm(coxa_vec)        
        coxa_vec = coxa_vec_unit*self.lengths[0]
        p1 = coxa_vec + start_p
        vec_p0_p1 = p1 - start_p
        alpha = vector_angle(coxa_vec_unit, np.array([0, 1, 0]))
        if coxa_vec[0] > 0:
            alpha *= -1
        rho = vector_angle(coxa_vec, vec_p0_p3, degree=False)
        if vec_p0_p3[-1] < 0:
            rho*=-1
        loc_p3y = vec_p0_p3_len * np.cos(rho)
        loc_p3z = vec_p0_p3_len * np.sin(rho)
        
        vec_p1_p3 = end_p - p1
        vec_p1_p3_len = np.linalg.norm(vec_p1_p3)
        
        if not can_form_triangle(vec_p1_p3_len, self.lengths[1], self.lengths[2]):
            # cannot reach the goal, so stretch the segments on the same line
            vec_p1_p2 = vec_p1_p3/vec_p1_p3_len*self.lengths[1]
            vec_p2_p3 = vec_p1_p3/vec_p1_p3_len*self.lengths[2]
            p2 = p1 + vec_p1_p2
            p3 = p2 + vec_p2_p3
            gamma = 0
            beta = vector_angle(vec_p0_p1, vec_p1_p2)
        else:
            # could form the triangle, use cosine theorem to get the angle between
            theta = np.arccos((vec_p1_p3_len**2 + self.lengths[1]**2 - self.lengths[2]**2)/(2*vec_p1_p3_len*self.lengths[1]))
            phi = vector_angle(vec_p1_p3, vec_p0_p1, degree=False)
            # different cases for the relationship of beta, phi and theta
            beta = theta - phi if loc_p3z < 0 else theta + phi
            loc_p2z = self.lengths[1]*np.sin(beta)
            loc_p2y = vec_p0_p1[1] + self.lengths[1]*np.cos(beta)
            vec_p1_p2 = np.array([0, loc_p2y, loc_p2z]) - vec_p0_p1
            p2 = p1 + vec_p1_p2
            vec_p2_p3 = np.array([0, loc_p3y, loc_p3z]) - np.array([0, loc_p2y, loc_p2z])
            p3 = p2 + vec_p2_p3
            gamma = vector_angle(vec_p2_p3, vec_p1_p2)
            if loc_p2z > loc_p3z:
                gamma *= -1
            beta = np.rad2deg(beta)
        
        diff = start_p - np.array(self.points_global[0].get_coordinates())
        self.global_transform = np.matmul(get_transformation_homo([0,0,0], diff), self.global_transform)
        
        # assign to the leg's attributes
        self.change_pose([alpha, beta, gamma])

        self.points_global[-1].set_coordinates(end_p)
        return alpha, beta, gamma
        
    def visualize3d(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d", proj_type="persp")
        # Points P0-04
        for i, p in enumerate(self.points_global):
            if i == 0:
                color='r'
            else:
                color='k'
            ax.scatter(p.x, p.y, p.z, s=100, color=color)
        for l in [(0,1),(1,2),(2,3)]:
            px = [self.points_global[l[0]].x, self.points_global[l[1]].x]
            py = [self.points_global[l[0]].y, self.points_global[l[1]].y]
            pz = [self.points_global[l[0]].z, self.points_global[l[1]].z]
            ax.plot(px, py, pz, lw=10, color='royalblue', alpha=0.6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        return fig, ax
    
    def visualizeAxis(self, axis='0123', scale=1, fig=None, ax=None):
        def plot_axis(ax, global_transform, transform, scale, text='axis'):
            px = [scale,0,0,1]
            py = [0,scale,0,1]
            pz = [0,0,scale,1]
            po = [0,0,0,1]
            px_t = global_transform.dot(transform.dot(px))
            py_t = global_transform.dot(transform.dot(py))
            pz_t = global_transform.dot(transform.dot(pz))
            po_t = global_transform.dot(transform.dot(po))
            
            xline, = ax.plot([po_t[0], px_t[0]], [po_t[1], px_t[1]], [po_t[2], px_t[2]], color='red', lw=2)
            yline, = ax.plot([po_t[0], py_t[0]], [po_t[1], py_t[1]], [po_t[2], py_t[2]], color='green', lw=2)
            zline, = ax.plot([po_t[0], pz_t[0]], [po_t[1], pz_t[1]], [po_t[2], pz_t[2]], color='blue', lw=2)
            ax.text(po_t[0], po_t[1], po_t[2], text)
            return ax
        
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        for plot_s in axis:
            ax = plot_axis(ax, self.global_transform, self.transforms[int(plot_s)], scale, text='P' + plot_s)
        ax = plot_axis(ax, np.identity(4), np.identity(4), scale, text='world')
        return fig, ax


class Hexapod:
    def __init__(self, leg_length=DEFAULT_LEG_LENGTH):
        self.body = Body(*DEFAULT_DIMSIONS)
        self.legs = {}
        self.leg_alpha_bias = DEFAULT_LEG_ALPHA_BIAS
        self.init_state(DEFAULT_DIMSIONS, leg_length)

        self.walking_sequence = {}
        self.generate_walking_sequence(dict(Gait='Tripod', HipSwing=30, LiftSwing=20, StepNum=6,
                                            Direction=1, Rotation=0))
        
    def init_state(self, body_dim, leg_length):
        self.initial_pose = {}
        self.body.init_vertices(body_dim)
        self.init_legs(leg_length)
        self.x_axis = Point3D(1, 0, 0, name='hexapod_x_axis')
        self.y_axis = Point3D(0, 1, 0, name='hexapod_y_axis')
        self.z_axis = Point3D(0, 0, 1, name='hexapod_z_axis')
        self.transform_m = np.identity(4)
        self.ground_contact_points = {}
        for k, v in self.legs.items():
            self.ground_contact_points[k] = v.get_ground_contact_point()
        self.ground_contact_points_old = deepcopy(self.ground_contact_points)
        self.body_plane_height = 0
        return self.update_state()
        
    def init_legs(self, lengths, init_gamma=DEFAULT_LEG_GAMMA):
        for k, v in LEG_ID_NAMES.items():
            self.legs[k] = Leg(lengths=lengths, name=v)
            self.legs[k].global_transform = get_transformation_homo([0, 0, 0], self.body.vertices[k].get_coordinates())
            self.legs[k].angles = [self.leg_alpha_bias[k], 0, init_gamma]
            self.initial_pose[v] = {'coxa':self.leg_alpha_bias[k],
                                    'femur':0,
                                    'tibia':init_gamma}
            self.legs[k]._update_transforms()
            self.legs[k]._update_pose()

    def get_legs_pose(self):
        pose = {}
        for leg_id, leg in self.legs.items():
            leg_dict = {}
            for k, v in LEG_SEG_NAMES_ID.items():
                leg_dict[k] = leg.angles[v] 
            pose[LEG_ID_NAMES[leg_id]] = leg_dict
        return pose
        
    def update_axis(self, transform=np.identity(4)):
        self.x_axis.set_coordinates(transform.dot(self.x_axis.get_coordinates_homo()))
        self.y_axis.set_coordinates(transform.dot(self.y_axis.get_coordinates_homo()))
        self.z_axis.set_coordinates(transform.dot(self.z_axis.get_coordinates_homo()))
    
    def transform(self, transform):
        self.transform_m = transform.dot(self.transform_m)
        self.body.transform(transform)
        for l in self.legs.values():
            l.transform(transform)
        self.update_axis(transform)
    
    def body_transform(self, transform):
        self.transform_m = transform.dot(self.transform_m)
        self.body.transform(transform)
        self.update_axis(transform)
        
    def update_leg_pose(self, poses:dict):
        '''poses is a dict of dicts in format: {"leg_name":{"leg_seg_name":x degree}}
        '''
        self.ground_contact_points_old = deepcopy(self.ground_contact_points)
        # determine if the hexapod should twist along z-axis due to 
        # the change of alpha angle of the legs on the ground
        body_might_twist = False
        cnt = 0
        for n in self.ground_contact_points:
            if n in poses.keys():
                if 'coxa' in poses[n]:
                    if abs(poses[n]['coxa'] - self.legs[n].angles[0]-self.initial_pose[n][0]) > 0:
                        cnt += 1
            if cnt >= 3:
                body_might_twist = True
                break
        
        for leg, v in poses.items():
            for leg_seg, angle in v.items():
                self.legs[LEG_NAMES_ID[leg]].angles[LEG_SEG_NAMES_ID[leg_seg]] = self.initial_pose[leg][leg_seg] + angle
            self.legs[LEG_NAMES_ID[leg]]._update_transforms()
            self.legs[LEG_NAMES_ID[leg]]._update_pose()
        
        return self.update_state(body_might_twist)
    
    def update_leg_pattern(self, angles):
        self.ground_contact_points_old = deepcopy(self.ground_contact_points)
        cnt = 0
        # all the legs share the same pose
        for i, l in enumerate(self.legs.values()):
            if abs(angles[0] - DEFAULT_LEG_ALPHA_BIAS[i] - l.angles[0]) > 0:
                cnt += 1
            l.angles = [angles[0] + DEFAULT_LEG_ALPHA_BIAS[i], angles[1], angles[2] + DEFAULT_LEG_GAMMA]
            l._update_transforms()
            l._update_pose()
        
        return self.update_state(body_might_twist=(cnt>=3))
        
    def update_dimensions(self, dimension):
        '''dimension is a list: [f, m, s, coxa, femur, tibia]
        '''
        return self.init_state(dimension[:3], dimension[3:])
    
    def update_state(self, body_might_twist=False):
        '''update the pose of the robot
        '''
        is_stable = False
        # find the ground contact point constructing the support polygon
        for leg_inds in LEG_TRIOS:
            p0, p1, p2 = [self.legs[i].get_ground_contact_point() for i in leg_inds]

            # justify if the cog is in the triangle formed by these legs
            if not is_projected_point_within_triangle(np.array(self.body.cog.get_coordinates()),
                                                      [np.array(p0.get_coordinates()),
                                                       np.array(p1.get_coordinates()),
                                                       np.array(p2.get_coordinates())]):
                continue
            n = get_plane_norm(np.array(p0.get_coordinates()),
                               np.array(p1.get_coordinates()),
                               np.array(p2.get_coordinates()))
            # get distance from the cog to the support polygon plane
            d = n.dot((self.body.cog - p0).get_coordinates())
            
            # check if this trio constructs the lowest plane (i.e., the biggest d)
            others = [self.legs[i].get_ground_contact_point() for i in set(LEG_ID_NAMES.keys()) - set(leg_inds)]
            r = True
            for p in others:
                d_ = n.dot((self.body.cog - p).get_coordinates())
                if d_ > d:
                   r = False
            if r:
                is_stable = True
                self.body_plane_norm = n
                break
        
        if is_stable:
            self.ground_contact_points = {}
            for leg_id in leg_inds:
                self.ground_contact_points[leg_id] = self.legs[leg_id].get_ground_contact_point()
            # get all the legs' end tip that contacts the ground
            for leg_id in set(LEG_ID_NAMES.keys()) - set(leg_inds):
                for p in reversed(self.legs[leg_id].points_global[1:]):
                    d_ = n.dot((self.body.cog - p).get_coordinates())
                    if np.isclose(d, d_):
                        self.ground_contact_points[leg_id] = p
                        break
            # tilt the hexapod according to the new plane norm
            rot_m = get_rotation_matrix_align_vectors(self.body_plane_norm, np.array([0, 0, 1]))
            t = combine_rot_trans_to_homo(rot_m, [0, 0, d - self.body_plane_height])
            self.transform(t)
            self.body_plane_height = d
            # twist body if needed
            if body_might_twist:
                # find one pair of point to get the twist angle
                for k, v in self.ground_contact_points_old.items():
                    if k in self.ground_contact_points:
                        a = np.arctan2(v.y, v.x)-np.arctan2(self.ground_contact_points[k].y, self.ground_contact_points[k].x)
                        t = get_transformation_homo([0,0,np.rad2deg(a)],[0,0,0])
                        self.transform(t)
                        break
            return True
        else:
            print('The pose is not stable, keep previous pose')
            return False

    def solve_ik(self, rot, trans):
        # reset hexapod
        self.init_state((self.body.f, self.body.m, self.body.s), self.legs[0].lengths)
        # restore old body contacts
        old_body_contacts = deepcopy(self.body.vertices)
        # transform body to get new body contacts
        self.body_transform(get_transformation_homo(rot, trans))
        # solve IK for each leg
        for k, leg in self.legs.items():
            leg.solve_ik(np.array(self.body.vertices[k].get_coordinates()),
                         np.array(self.ground_contact_points[k].get_coordinates()))
    
    def generate_walking_sequence(self, parameters:dict):
        gait = parameters['Gait']
        d_alpha = parameters['HipSwing']
        d_beta = parameters['LiftSwing']
        d_gamma = -parameters['LiftSwing']/2
        step_num = parameters['StepNum']
        move_dir = parameters['Direction']
        rotation = parameters['Rotation']
        self.init_state((self.body.f, self.body.m, self.body.s), self.legs[0].lengths)
        self.walking_sequence = {}
        if gait == 'Tripod':
            for k, leg in self.legs.items():
                beta_s = np.linspace(0, d_beta, int(step_num))
                beta_s_r = beta_s[::-1]
                beta_s_0 = np.ones(int(len(beta_s)*2))*beta_s[0]
                gamma_s = np.linspace(0, d_gamma, int(step_num))
                gamma_s_r = gamma_s[::-1]
                gamma_s_0 = np.ones(int(len(gamma_s)*2))*gamma_s[0]
                if rotation != 0:
                    alpha_s = np.linspace(-d_alpha, d_alpha, int(step_num*2))
                    alpha_s_r = alpha_s[::-1]
                else:
                    if k in (2, 3, 4):
                        alpha_s = np.linspace(d_alpha, -d_alpha, int(step_num*2))
                        alpha_s_r = alpha_s[::-1]
                    else:
                        alpha_s = np.linspace(-d_alpha, d_alpha, int(step_num*2))
                        alpha_s_r = alpha_s[::-1]
                alpha_seq_a = np.hstack([alpha_s, alpha_s_r])
                alpha_seq_b = np.hstack([alpha_s_r, alpha_s])
                if k in (0, 2, 4):
                    self.walking_sequence[k] = {'coxa':alpha_seq_a,
                                                'femur':np.hstack([beta_s, beta_s_r, beta_s_0]),
                                                'tibia':np.hstack([gamma_s, gamma_s_r, gamma_s_0])}
                else:
                    self.walking_sequence[k] = {'coxa':alpha_seq_b,
                                                'femur':np.hstack([beta_s_0, beta_s, beta_s_r]),
                                                'tibia':np.hstack([gamma_s_0, gamma_s, gamma_s_r])}
        elif gait == 'Ripple':
            pass
    
    def set_pose_from_walking_sequence(self, step):
        poses = {}
        for k, v in self.walking_sequence.items():
            P = {}
            for seg in LEG_SEG_NAMES_ID.keys():
                P[seg] = v[seg][step]
            poses[LEG_ID_NAMES[k]] = P
        return self.update_leg_pose(poses)
        
        
    def visualize3d(self, fig=None, ax=None):
        if fig is None:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
        fig, ax = self.body.visualize3d(fig, ax)
        for k, leg in self.legs.items():
            fig, ax = leg.visualize3d(fig, ax)
        # hexapod axis
        ax.plot([self.body.cog.x, self.x_axis.x],
                [self.body.cog.y, self.x_axis.y],
                [self.body.cog.z, self.x_axis.z],
                color='red')
        ax.plot([self.body.cog.x, self.y_axis.x],
                [self.body.cog.y, self.y_axis.y],
                [self.body.cog.z, self.y_axis.z],
                color='green')
        ax.plot([self.body.cog.x, self.z_axis.x],
                [self.body.cog.y, self.z_axis.y],
                [self.body.cog.z, self.z_axis.z],
                color='blue')
        ax.set_aspect('equal')
        ax.set_zlim([0,6])
        return fig, ax


if __name__ == "__main__":
    # hexapod = Hexapod()
    # fig1, ax1 = hexapod.visualize3d()
    # hexapod.solve_ik([0, 6, 2], [0, 0, 0.2])
    # print(hexapod.get_legs_pose())
    # fig2, ax2 = hexapod.visualize3d()
    # plt.show()

    # leg = Leg(lengths=[2,2,3])
    # t = get_transformation_homo([0, 0, 45], [0,4,0])
    # leg.change_pose([-60, 45, 30])
    # leg.transform(t)
    # fig, ax = leg.visualize3d()
    # fig, ax = leg.visualizeAxis(fig=fig, ax=ax, scale=1)
    # plt.show()
    pass