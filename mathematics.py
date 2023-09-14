import numpy as np
from scipy.spatial.transform import Rotation as R

def combine_rot_trans_to_homo(rot, trans):
    M = np.identity(4)
    M[:3, :3] = rot
    M[:3, 3] = trans
    return M

def get_transformation_homo(rot, trans, degrees=True):
    M = np.identity(4)
    M[:3, :3] = R.from_euler("XYZ", rot, degrees=degrees).as_matrix()
    # translation
    M[:3, 3] = trans
    return M

def get_plane_norm(p0, p1, p2):
    '''get the unit vector of the plane's norm given three points
    '''
    u = p1 - p0
    v = p2 - p0
    n = np.cross(u, v)
    n = n / np.linalg.norm(n)
    return n

def is_point_within_triangle_same_plane(point, triangle):
    '''determine if the point is within the triangle formed by three points
    point is already in the plane formed defined by the triangle
    '''
    AB = triangle[1]-triangle[0]
    AC = triangle[2]-triangle[0]
    PA = point - triangle[0]
    PB = point - triangle[1]
    PC = point - triangle[2]
    area_triangle = np.linalg.norm(np.cross(AB, AC))
    alpha = np.linalg.norm(np.cross(PB, PC))/area_triangle
    beta = np.linalg.norm(np.cross(PC, PA))/area_triangle
    gamma = 1 - alpha - beta
    return (0 < alpha < 1) and (0 < beta < 1) and (0 < gamma < 1)

# https://math.stackexchange.com/questions/544946/determine-if-projection-of-3d-point-onto-plane-is-within-a-triangle
# https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
# https://en.wikipedia.org/wiki/Barycentric_coordinate_system
def is_projected_point_within_triangle(point, triangle):
    '''determine if the point is within the triangle formed by three points
    by applying the Barycentric coordinates
    '''
    u = triangle[1]-triangle[0]
    v = triangle[2]-triangle[0]
    w = point - triangle[0]
    n = np.cross(u, v)
    alpha = np.linalg.norm(np.cross(u, w).dot(n))/n.dot(n)
    beta = np.linalg.norm(np.cross(w, v).dot(n))/n.dot(n)
    gamma = 1 - alpha - beta
    return (0 < alpha < 1) and (0 < beta < 1) and (0 < gamma < 1)

# https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
# https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
def get_rotation_matrix_align_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0.0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

# https://www.maplesoft.com/support/help/Maple/view.aspx?path=MathApps%2FProjectionOfVectorOntoPlane
def project_vector_onto_plane(vec, plane_norm):
    s = vec.dot(plane_norm) / plane_norm.dot(plane_norm)
    return vec - s*plane_norm

def vector_angle(v1, v2, degree=True):
    v = v1.dot(v2)/np.sqrt(v1.dot(v1)*v2.dot(v2))
    if abs(v) <= 1:
        a = np.arccos(v)
    else:
        a = 0.0
    
    return a if not degree else np.rad2deg(a)

def can_form_triangle(a, b, c):
    return (a + b > c) and (a + c > b) and (b + c > a)

class Point3D:
    def __init__(self, x=0, y=0, z=0, name='None') -> None:
        self.x = x
        self.y = y
        self.z = z
        self.name = name

    def get_coordinates(self):
        return self.x, self.y, self.z

    def set_coordinates(self, coordinates):
        if hasattr(coordinates, '__len__'):
            if len(coordinates) >= 3:
                self.x = coordinates[0]
                self.y = coordinates[1]
                self.z = coordinates[2]
            else:
                raise ValueError
        else:
            raise ValueError

    def get_coordinates_homo(self):
        return self.x, self.y, self.z, 1

    def __repr__(self):
        s = f"{self.name}:({self.x:>4.2f}, {self.y:>4.2f}, {self.z:>4.2f})"
        return s

    def __add__(self, b):
        if isinstance(b, Point3D):
            return Point3D(self.x + b.x, self.y + b.y, self.z + b.z, name=self.name)
        elif isinstance(b, float) or isinstance(b, int):
            return Point3D(self.x + b, self.y + b, self.z + b, name=self.name)
        elif hasattr(b, "__len__"):
            if len(b) == 3:
                return Point3D(self.x + b[0], self.y + b[1], self.z + b[2], name=self.name)
        else:
            raise ValueError
    
    def __sub__ (self, b):
        if isinstance(b, Point3D):
            return Point3D(self.x - b.x, self.y - b.y, self.z - b.z, name=self.name + '-' + b.name)
        elif isinstance(b, float) or isinstance(b, int):
            return Point3D(self.x - b, self.y - b, self.z - b, name=self.name + '-' + str(b))
        else:
            raise ValueError
    
    def dot(self, b):
        if isinstance(b, Point3D):
            return self.x*b.x + self.y*b.y + self.z*b.z
        else:
            raise ValueError