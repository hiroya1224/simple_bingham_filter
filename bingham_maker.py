import numpy as np

class Quaternion:
    def __init__(self, w, x, y, z):
        self._array = np.array([w, x, y, z])
        self.w = w
        self.x = x
        self.y = y
        self.z = z
    
    def array(self):
        return self._array
    
    def real(self):
        return self.w
    
    def imag(self):
        return np.array([self.x, self.y, self.z])
    
    def rotmat(self):
        w, x, y, z = self._array
        n11 = w**2 + x**2 - y**2 - z**2
        n21 = 2*(x*y + w*z)
        n31 = 2*(x*z - w*y)

        n12 = 2*(x*y - w*z)
        n22 = w**2 - x**2 + y**2 - z**2
        n32 = 2*(y*z + w*x)

        n13 = 2*(x*z + w*y)
        n23 = 2*(y*z - w*x)
        n33 = w**2 - x**2 - y**2 + z**2

        return np.array([[n11, n12, n13], [n21, n22, n23], [n31, n32, n33]])


def Lmat(quat):
    a, b, c, d = quat.array()
    return np.array([[a, -b, -c, -d], [b, a, -d, c], [c, d, a, -b], [d, -c, b, a]])

def Rmat(quat):
    w, x, y, z = quat.array()
    return np.array([[w, -x, -y, -z], [x, w, z, -y], [y, -z, w, x], [z, y, -x, w]])

def pseudo_measurement_matrix(before_rotation_3dvec, after_rotation_3dvec):
    vq = Quaternion(0, *before_rotation_3dvec)
    xq = Quaternion(0, *after_rotation_3dvec)
    return Lmat(xq) - Rmat(vq)


x = (lambda x: x / np.linalg.norm(x))(np.random.randn(3))
q = (lambda x: x / np.linalg.norm(x))(np.random.randn(4))
R = Quaternion(*q).rotmat()

y = np.dot(R, x)
P  = pseudo_measurement_matrix(x, R @ x)


def simple_bingham_unit(before_rotation_3dvec, after_rotation_3dvec, parameter=100.):
    vq = Quaternion(0, *before_rotation_3dvec)
    xq = Quaternion(0, *after_rotation_3dvec)
    P  = Lmat(xq) - Rmat(vq)
    # maximum eigenvalue of P.T @ P is 4.0, so 0.25 is set for making it 1.
    A0 = -0.25 * P.T @ P
    return parameter * A0

# def axis_symmetric_bingham(rotational_axis_3dvec):
#     # Note that L_matrix and R_matrix are commutable.
#     K = Lmat(left_quat) @ Rmat(right_quat)
#     return K @ A @ K.T

# print(P)
# print(q)
# print(np.linalg.eigh(simple_bingham_unit(x, R @ x)))

# print(P @ q)

c = np.dot(x,y)

v = np.array([0, 1, 0])
# v = (lambda x: x / np.linalg.norm(x))(np.random.randn(3))

def parametrize_Qv(alpha, beta, v, theta):
    Pv = simple_bingham_unit(v, v, parameter=1.)
    qv = Quaternion(np.cos(theta), *(v * np.sin(theta)))
    Qv = np.outer(qv.array(), qv.array())

    alpha = np.clip(alpha, 1e-6, np.inf) / 1000
    beta = np.clip(beta, 1e-6, np.inf) / 1000
    return Pv / alpha + (Qv - np.eye(4)) / beta

Qv = parametrize_Qv(0.5, 20, v, 0.2)

print(np.linalg.eigh(Qv))