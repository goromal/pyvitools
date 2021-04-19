## @package pyvitools
#  Select tools written in Python for processing and analyzing data for visual-inertial estimation applications.
#
#  Provides tools for:
#
#   * Algebraic anipulation of vectors and SO(3)/SE(3) objects.
#   * Differentiating and integrating the above.
#   * Gaussian statistics analysis.
#   * ROSbag processing.
#   * Dataset manipulation.
#   * (Coming soon) Plotting of all the above.

# Base packages
import numpy as np
import csv
from math import sqrt, atan2, cos, sin, pi, asin
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import product
from shapely.geometry import MultiPoint

# Import opencv if available
try:
    import cv2
    __CV = True
except ImportError:
    print('NO OPENCV')
    __CV = False

# Import point cloud visualization if available
try:
    import open3d
    _PCL_vis = True
except ImportError:
    print('NO POINTCLOUD VIS')
    _PCL_vis = False

# Import ROS message types if available
try:
    from geometry_msgs.msg import Quaternion, QuaternionStamped
    from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped
    from geometry_msgs.msg import Transform, TransformStamped
    from geometry_msgs.msg import Point, PointStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image, Imu
    _ROS_msgs = True
except ImportError:
    print('NO ROS MSGS')
    _ROS_msgs = False

# Import ROSbag parser if available
try:
    import rospy, rosbag
    _ROS_bag = True
except ImportError:
    print('NO ROSBAG PARSING')
    _ROS_bag = False

# Import ROS cv bridge if available
try:
    from cv_bridge import CvBridge
    _ROS_bridge = True
except ImportError:
    print('NO CV BRIDGE')
    _ROS_bridge = False

class ROSNotFoundError(Exception):
    pass

##################################################################
######################## MANIFOLD OBJECTS ########################

## Convert quaternion to rotation matrix.
#  @param q \f$4\times 1\f$ numpy array \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
#  @return \f$3\times 3\f$ numpy array.
def q2R(q):
    qw, qx, qy, qz = q[0,0], q[1,0], q[2,0], q[3,0]
    qxx = qx*qx
    qxy = qx*qy
    qxz = qx*qz
    qyy = qy*qy
    qyz = qy*qz
    qzz = qz*qz
    qwx = qw*qx
    qwy = qw*qy
    qwz = qw*qz
    return np.array([[1-2*qyy-2*qzz, 2*qxy-2*qwz, 2*qxz+2*qwy],
                     [2*qxy+2*qwz, 1-2*qxx-2*qzz, 2*qyz-2*qwx],
                     [2*qxz-2*qwy, 2*qyz+2*qwx, 1-2*qxx-2*qyy]])

## Convert rotation matrix to quaternion.
#  @param R \f$3\times 3\f$ numpy array.
#  @return \f$4\times 1\f$ numpy array \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
def R2q(R):
    R11, R12, R13 = R[0,0], R[0,1], R[0,2]
    R21, R22, R23 = R[1,0], R[1,1], R[1,2]
    R31, R32, R33 = R[2,0], R[2,1], R[2,2]

    if R11 + R22 + R33 > 0.0:
        s = 2 * sqrt(1 + R11 + R22 + R33)
        qw = s/4
        qx = 1/s*(R32-R23)
        qy = 1/s*(R13-R31)
        qz = 1/s*(R21-R12)
    elif R11 > R22 and R11 > R33:
        s = 2 * sqrt(1 + R11 - R22 - R33)
        qw = 1/s*(R32-R23)
        qx = s/4
        qy = 1/s*(R21+R12)
        qz = 1/s*(R31+R13)
    elif R22 > R33:
        s = 2 * sqrt(1 + R22 - R11 - R33)
        qw = 1/s*(R13-R31)
        qx = 1/s*(R21+R12)
        qy = s/4
        qz = 1/s*(R32+R23)
    else:
        s = 2 * sqrt(1 + R33 - R11 - R22)
        qw = 1/s*(R21-R12)
        qx = 1/s*(R31+R13)
        qy = 1/s*(R32+R23)
        qz = s/4
        
    q = np.array([[qw, qx, qy, qz]]).T

    if q[0,0] < 0:
        q *= -1

    return q 

def qL(q):
    qw, qx, qy, qz = q[0,0], q[1,0], q[2,0], q[3,0]
    return np.array([[qw, -qx, -qy, -qz],
                     [qx, qw, -qz, qy],
                     [qy, qz, qw, -qx],
                     [qz, -qy, qx, qw]])

def xyz_msg2array(msg):
    return np.array([[msg.x],[msg.y],[msg.z]])

class SO3(object):
    @staticmethod
    def random():
        return SO3(np.random.random((4,1)))

    @staticmethod
    def identity():
        return SO3(qw=1.0, qx=0, qy=0, qz=0)

    @staticmethod
    def fromQuaternionMsg(msg):
        if not _ROS_msgs:
            raise ROSNotFoundError('ROS message types not found on system.')
        return SO3(qw=msg.w, qx=msg.x, qy=msg.y, qz=msg.z)

    @staticmethod
    def fromRotationMatrix(R):
        return SO3(R2q(R))

    @staticmethod
    def fromQuaternion(q):
        return SO3(q)

    @staticmethod
    def fromTwoUnitVectors(u, v):
        d = np.dot(u.T, v)
        if d < 0.99999999 and d > -0.99999999:
            invs = 1.0 / sqrt((2.0*(1.0+d)))
            xyz = np.cross(u, v*invs, axis=0)
            return SO3(qw=0.5/invs, qx=xyz[0,0], qy=xyz[1,0], qz=xyz[2,0])
        elif d < -0.99999999:
            return SO3(qw=0., qx=0., qy=1., qz=0.)
        else:
            return SO3(qw=1., qx=0., qy=0., qz=0.)

    def __init__(self, arr=None, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
        if arr is None:
            self.arr = np.array([[qw, qx, qy, qz]]).T
        else:
            if not isinstance(arr, np.ndarray) and arr.shape != (4,1):
                raise TypeError('SO(3) default constructor requires a 4x1 numpy array quaternion [qw qx qy qz]^T.')
            self.arr = arr 
        self.arr /= np.linalg.norm(self.arr)
        if self.arr[0,0] < 0:
            self.arr *= -1

    def R(self):
        return q2R(self.arr)

    def q(self):
        return self.arr

    def inverse(self):
        return SO3(qw=self.arr[0,0], qx=-self.arr[1,0], qy=-self.arr[2,0], qz=-self.arr[3,0])

    def invert(self):
        self.arr[1:,:] *= -1.0

    def toQuaternionMsg(self):
        if not _ROS_msgs:
            raise ROSNotFoundError('ROS message types not found on system.')
        msg = Quaternion()
        msg.w = self.arr[0,0]
        msg.x = self.arr[1,0]
        msg.y = self.arr[2,0]
        msg.z = self.arr[3,0]
        return msg

    ## Convert SO3 to Euler representation (roll, pitch, yaw).
    #
    #  Uses the flight dynamics convention: roll about x -> pitch about y -> yaw about z (http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles)
    def toEuler(self):
        qw = self.arr[0,0]
        qx = self.arr[1,0]
        qy = self.arr[2,0]
        qz = self.arr[3,0]

        yr = 2.*(qw*qx + qy*qz)
        xr = 1 - 2*(qx**2 + qy**2)
        roll = atan2(yr, xr)

        sp = 2.*(qw*qy - qz*qx)
        pitch = asin(sp)

        yy = 2.*(qw*qz + qx*qy)
        xy = 1 - 2.*(qy**2 + qz**2)
        yaw = atan2(yy, xy)

        return roll, pitch, yaw

    def __mul__(self, other):
        if not isinstance(other, SO3):
            raise TypeError('SO(3) multiplication is only valid with another SO(3) object.')
        return SO3(np.dot(qL(self.arr), other.arr))

    def __add__(self, other):
        if isinstance(other, np.ndarray) and other.shape == (3,1):
            return self * SO3.Exp(other)
        else:
            raise TypeError('SO(3) can only be perturbed by a 3x1 numpy array tangent space vector.')

    def __sub__(self, other):
        if isinstance(other, SO3):
            return SO3.Log(other.inverse() * self)
        else:
            raise TypeError('SO(3) can only be differenced by another SO(3) object.')

    @staticmethod
    def hat(o):
        if isinstance(o, np.ndarray) and o.shape == (3,1):
            return np.array([[0, -o[2,0], o[1,0]],
                             [o[2,0], 0, -o[0,0]],
                             [-o[1,0], o[0,0], 0]])
        else:
            raise TypeError('The SO(3) hat operator must take in a 3x1 numpy array tangent space vector.')

    @staticmethod
    def vee(Omega):
        if isinstance(Omega, np.ndarray) and Omega.shape == (3,3) and Omega.T == -Omega:
            return np.array([[Omega[2,1]],[Omega[0,2]],[Omega[1,0]]])
        else:
            raise TypeError('The SO(3) vee operator must take in a member of the so(3) Lie algebra.')

    @staticmethod
    def log(X):
        return SO3.hat(SO3.Log(X))

    @staticmethod
    def Log(X):
        qv = X.arr[1:,:]
        qw = X.arr[0,0]
        n = np.linalg.norm(qv)
        if n > 0.0:
            return qv * 2.0 * atan2(n, qw) / n
        else:
            return np.zeros((3,1))

    @staticmethod
    def exp(v):
        return SO3.Exp(SO3.vee(v))

    @staticmethod
    def Exp(v):
        th = np.linalg.norm(v)
        q = np.zeros((4,1))
        if th > 0.0:
            u = v / th
            q[0] = cos(th/2)
            q[1:,:] = u * sin(th/2)
        else:
            q[0] = 1.0
        return SO3(q)

class SE3(object):
    @staticmethod
    def random():
        return SE3(np.random.random((7,1)))

    @staticmethod
    def identity():
        return SE3(tx=0, ty=0, tz=0, qw=1.0, qx=0, qy=0, qz=0)

    @staticmethod
    def fromPoseMsg(msg):
        if not _ROS_msgs:
            raise ROSNotFoundError('ROS message types not found on system.')
        return SE3(tx=msg.position.x,
                   ty=msg.position.y,
                   tz=msg.position.z,
                   qw=msg.orientation.w,
                   qx=msg.orientation.x,
                   qy=msg.orientation.y,
                   qz=msg.orientation.z)

    @staticmethod
    def fromTransformMsg(msg):
        if not _ROS_msgs:
            raise ROSNotFoundError('ROS message types not found on system.')
        return SE3(tx=msg.translation.x,
                   ty=msg.translation.y,
                   tz=msg.translation.z,
                   qw=msg.rotation.w,
                   qx=msg.rotation.x,
                   qy=msg.rotation.y,
                   qz=msg.rotation.z)

    @staticmethod
    def fromTransformMatrix(T):
        return SE3(np.vstack((T[0:3,3:], R2q(T[0:3,0:3]))))

    @staticmethod
    def fromcSE3(v):
        return SE3(tx=v[0], ty=v[1], tz=v[2], qw=v[6], qx=v[3], qy=v[4], qz=v[5])

    @staticmethod
    def fromTranslationAndRotation(*args):
        if len(args) == 1:
            return SE3(args[0])
        elif len(args) == 2 and isinstance(args[1], np.ndarray):
            return SE3(np.vstack((args[0], args[1])))
        elif len(args) == 2 and isinstance(args[1], SO3):
            return SE3(np.vstack((args[0], args[1].q())))
        else:
            raise TypeError('Translation/Quaternion allowed types are (7x1 array), (3x1 array, 4x1 array), (3x1 array, SO3).')

    @staticmethod
    def fromCSVRow(row, start_idx=0):
        tx = float(row[start_idx+0])
        ty = float(row[start_idx+1])
        tz = float(row[start_idx+2])
        qw = float(row[start_idx+3])
        qx = float(row[start_idx+4])
        qy = float(row[start_idx+5])
        qz = float(row[start_idx+6])
        return SE3(tx=tx, ty=ty, tz=tz, qw=qw, qx=qx, qy=qy, qz=qz)

    def __init__(self, arr=None, tx=0, ty=0, tz=0, qw=1.0, qx=0, qy=0, qz=0):
        if arr is None:
            self.t = np.array([[tx, ty, tz]]).T
            self.q = SO3(qw=qw, qx=qx, qy=qy, qz=qz)
        else:
            if not isinstance(arr, np.ndarray) and arr.shape != (7,1):
                raise TypeError('SE(3) default constructor requires a 7x1 numpy array [tx ty tz qw qx qy qz]^T.')
            self.t = arr[0:3,:]
            self.q = SO3(arr[3:7,:])

    def T(self):
        return np.vstack((np.hstack((self.q.R(), self.t)), np.array([[0,0,0,1.0]])))

    def tq(self):
        return np.vstack((self.t, self.q.q()))

    def tqlist(self):
        return list((self.tq().T)[0])

    def tocSE3(self):
        return np.array([self.t[0,0], self.t[1,0], self.t[2,0], self.q.arr[1,0], self.q.arr[2,0], self.q.arr[3,0], self.q.arr[0,0]])

    def transform(self, v):
        return np.dot(self.q.R(), v) + self.t

    def inverse(self):
        return SE3.fromTranslationAndRotation(-np.dot(self.q.inverse().R(), self.t), self.q.inverse())

    def invert(self):
        self.q.invert()
        self.t = -np.dot(self.q.R(), self.t)

    def toPoseMsg(self):
        if not _ROS_msgs:
            raise ROSNotFoundError('ROS message types not found on system.')
        msg = Pose()
        msg.position.x = self.t[0,0]
        msg.position.y = self.t[1,0]
        msg.position.z = self.t[2,0]
        msg.orientation.w = self.q.arr[0,0]
        msg.orientation.x = self.q.arr[1,0]
        msg.orientation.y = self.q.arr[2,0]
        msg.orientation.z = self.q.arr[3,0]
        return msg

    def toTransformMsg(self):
        if not _ROS_msgs:
            raise ROSNotFoundError('ROS message types not found on system.')
        msg = Transform()
        msg.translation.x = self.t[0,0]
        msg.translation.y = self.t[1,0]
        msg.translation.z = self.t[2,0]
        msg.rotation.w = self.q.arr[0,0]
        msg.rotation.x = self.q.arr[1,0]
        msg.rotation.y = self.q.arr[2,0]
        msg.rotation.z = self.q.arr[3,0]
        return msg
    
    def __mul__(self, other):
        if not isinstance(other, SE3):
            raise TypeError('SE(3) multiplication is only valid with another SE(3) object.')
        return SE3.fromTranslationAndRotation(self.t + np.dot(self.q.R(), other.t), self.q * other.q)

    ## \f$\oplus\f$
    #
    # More details.
    def __add__(self, other):
        if isinstance(other, np.ndarray) and other.shape == (6,1):
            return self * SE3.Exp(other)
        else:
            raise TypeError('SE(3) can only be perturbed by a 6x1 numpy array tangent space vector.')

    def __sub__(self, other):
        if isinstance(other, SE3):
            return SE3.Log(other.inverse() * self)
        else:
            raise TypeError('SE(3) can only be differenced by another SE(3) object.')

    @staticmethod
    def hat(o):
        if isinstance(o, np.ndarray) and o.shape == (6,1):
            return np.vstack((np.hstack((SO3.hat(o[3:,:]),o[:3,:])),np.zeros((1,4))))
        else:
            raise TypeError('The SE(3) hat operator must take in a 6x1 numpy array tangent space vector.')

    @staticmethod
    def vee(Omega):
        if isinstance(Omega, np.ndarray) and Omega.shape == (4,4):
            return np.vstack((Omega[:3,3:], SO3.vee(Omega[:3,:3])))
        else:
            raise TypeError('The SE(3) vee operator must take in a member of the se(3) Lie algebra.')

    @staticmethod
    def log(X):
        return SE3.hat(SE3.Log(X))

    @staticmethod
    def Log(X):
        w = SO3.Log(X.q)
        th = np.linalg.norm(w)
        W = SO3.hat(w)

        if th > 0:
            a = sin(th)/th
            b = (1-cos(th))/(th**2)
            c = (1-a)/(th**2)
            e = (b-2*c)/(2*a)
            Jl_inv = np.eye(3) - 0.5 * W + e * np.dot(W, W)
        else:
            Jl_inv = np.eye(3)

        return np.vstack((np.dot(Jl_inv, X.t), w))
        # return np.vstack((X.t, w))

    @staticmethod
    def exp(v):
        return SE3.Exp(SE3.vee(v))

    @staticmethod
    def Exp(v):
        rho = v[:3,:]
        w = v[3:,:]
        q = SO3.Exp(w)
        W = SO3.hat(w)
        th = np.linalg.norm(w)

        if th > 0:
            a = sin(th)/th
            b = (1-cos(th))/(th**2)
            c = (1-a)/(th**2)
            Jl = a * np.eye(3) + b * W + c * np.dot(w, w.T)
        else:
            Jl = np.eye(3)

        return SE3.fromTranslationAndRotation(np.dot(Jl, rho), q)

def asFlatNPArray(v):
    return np.array(v.T[0])

def asVertNPArray(v):
    return np.array([v]).T

##################################################################
######################### SIGNALS TOOLS ##########################

# https://github.com/goromal/math-utils-lib/blob/master/include/math-utils-lib/modeling.h

class Differentiator(object):
    def __init__(self, shape, sigma=0.05):
        self.shape = shape
        self.sigma = sigma
        self.deriv_curr = None
        self.deriv_prev = None
        self.val_prev   = None
        self.initialized = False

    def calculate(self, val, Ts):
        if self.initialized:
            self.deriv_curr = (2 * self.sigma - Ts) / (2 * self.sigma + Ts) * self.deriv_prev + \
                              2 / (2 * self.sigma + Ts) * (val - self.val_prev)
        else:
            self.deriv_curr = np.zeros(self.shape)
            self.initialized = True
        self.deriv_prev = self.deriv_curr
        self.val_prev = val
        return self.deriv_curr

# https://github.com/goromal/matlab_utilities/blob/master/math/GeneralizedInterpolator.m

class InterpolatorBase(object):
    # list and list
    def __init__(self, t_data, y_data):
        self.t_data = t_data
        self.n = len(t_data)
        self.y_data = y_data
        self.i = 0

    def y(self, t):
        if t < self.t_data[0]:
            return self.y_data[0]
        if t > self.t_data[-1]:
            return self.y_data[-1]
        if t in self.t_data:
            return self.y_data[self.t_data.index(t)]
        for idx in range(self.n-1):
            self.i = idx
            ti = self.t_data[self.i]
            if ti < t and self._ti(self.i+1) > t:
                return self._interpy(t)
        return None

    def _interpy(self, t):
        return self._yi(self.i) + self._dy(t)

    def _dy(self, t):
        return None

    def _yi(self, i):
        if i < 0:
            return self.y_data[0]
        elif i >= self.n:
            return self.y_data[-1]
        else:
            return self.y_data[i]

    def _ti(self, i):
        if i < 0:
            return self.t_data[0] - 1.0
        elif i >= self.n:
            return self.t_data[-1] + 1.0
        else:
            return self.t_data[i]

class ZeroOrderInterpolator(InterpolatorBase):
    def __init__(self, t_data, y_data, zero_obj):
        super(ZeroOrderInterpolator, self).__init__(t_data, y_data)
        self.zero_obj = zero_obj

    def _dy(self, t):
        return self.zero_obj

class LinearInterpolator(InterpolatorBase):
    def __init__(self, t_data, y_data):
        super(LinearInterpolator, self).__init__(t_data, y_data)

    def _dy(self, t):
        t1 = self._ti(self.i)
        t2 = self._ti(self.i+1)
        y1 = self._yi(self.i)
        y2 = self._yi(self.i+1)
        return (t - t1) / (t2 - t1) * (y2 - y1)

class SplineInterpolator(InterpolatorBase):
    def __init__(self, t_data, y_data):
        super(SplineInterpolator, self).__init__(t_data, y_data)

    def _dy(self, t):
        t0 = self._ti(self.i-1)
        t1 = self._ti(self.i)
        t2 = self._ti(self.i+1)
        t3 = self._ti(self.i+2)
        y0 = self._yi(self.i-1)
        y1 = self._yi(self.i)
        y2 = self._yi(self.i+1)
        y3 = self._yi(self.i+2)
        return (t-t1)/(t2-t1)*((y2-y1) + \
             (t2-t)/(2*(t2-t1)**2)*(((t2-t)*(t2*(y1-y0)+t0*(y2-y1)-t1*(y2-y0)))/(t1-t0) + \
             ((t-t1)*(t3*(y2-y1)+t2*(y3-y1)-t1*(y3-y2)))/(t3-t2)))

##################################################################
########################## STATS TOOLS ###########################

## Returns (and optionally visualizes) the zero-mean standard deviation of the error between noisy_data and true_data. Assumes that noisy_data can be considered as true_data plus added Gaussian noise.
#  @param error_data Numpy array of length \f$n\f$, equal to noisy data - true data.
#  @param covariance_gating Whether or not to attempt to reject outliers via single-pass covariance gating.
#  @param title (Optional) Title for report and/or plot.
#  @param report_stats (Optional) Report diagnostics evaluating Gaussian assumption suitability.
#  @param plot (Optional) Visualize error probability distribution and Gaussian fit.
def getZeroMeanGaussianErrorStdDev(error_data, covariance_gating=False, title='Zero-Mean Gaussian Error Fit', report_stats=True, plot=False):
    n = error_data.size
    stdev = np.linalg.norm(error_data) / sqrt(n)

    if covariance_gating:
        outlierless_data = list()
        for i in range(n):
            if abs(error_data[i]) <= 3.0*stdev:
                outlierless_data.append(error_data[i])
        stdev = np.linalg.norm(np.array(outlierless_data)) / sqrt(1.0*len(outlierless_data))

    if report_stats:
        num_gzero = 0
        num_1sig = 0
        num_2sig = 0
        num_3sig = 0
        for i in range(n):
            val = error_data[i]
            if val >= 0:
                num_gzero += 1
            if abs(val) <= stdev:
                num_1sig += 1
            if abs(val) <= 2*stdev:
                num_2sig += 1
            if abs(val) <= 3*stdev:
                num_3sig += 1
        report_str = """{}:
    ZERO-MEAN STDEV: {}
               >= 0: {:.1f}% (vs 50.0%)
            1-SIGMA: {:.1f}% (vs 68.0%)
            2-SIGMA: {:.1f}% (vs 95.0%)
            3-SIGMA: {:.1f}% (vs 99.7%)
    """.format(title, stdev, 100.0*num_gzero/n, 100.0*num_1sig/n, 100.0*num_2sig/n, 100.0*num_3sig/n)
        print(report_str)

    if plot:
        x = np.linspace(-3*stdev, 3*stdev, 100)
        counts, bins = np.histogram(error_data, bins=40)
        plt.hist(bins[:-1], bins, weights=counts, density=True)
        plt.plot(x, stats.norm.pdf(x, 0.0, stdev))
        plt.title(title)
        plt.show()

    return stdev

##################################################################
########################### POINTCLOUD ###########################

def visualizePLYFile(filename):
    if not _PCL_vis:
        raise Exception('open3d not installed. If using Python3, install with pip or other package manager. Otherwise, build from source! See README.md.')
    print('Loading PLY file...')
    cloud = open3d.io.read_point_cloud(filename)
    print('Visualizing...')
    mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
    open3d.visualization.draw_geometries([cloud, mesh_frame])

##################################################################
####################### VECTOR SPACE TOOLS #######################

def projMatFromNormal(n):
    return np.eye(3) - np.dot(n, n.T)

def squashTo2DPlane(v, n):
    R = SO3.fromTwoUnitVectors(np.array([[0.,0.,1.]]).T, n).inverse().R()
    P_2D = np.dot(R, projMatFromNormal(n))
    return np.dot(P_2D, v)[0:2,:]

##################################################################
######################### PINHOLE TOOLS ##########################

def focal2fov(w, h, fx, fy):
    return (2 * atan2(w, 2 * fx), 2 * atan2(h, 2 * fy))

def cam3DPolyPoints(T, fov_x, fov_y, max_depth):
    pB = np.array([[0.0, max_depth, max_depth, max_depth, max_depth],
                   [0.0, max_depth * sin(fov_x / 2.0), max_depth * sin(fov_x / 2.0), -max_depth * sin(fov_x / 2.0), -max_depth * sin(fov_x / 2.0)],
                   [0.0, max_depth * sin(fov_y / 2.0), -max_depth * sin(fov_y / 2.0), max_depth * sin(fov_y / 2.0), -max_depth * sin(fov_y / 2.0)]])
    return T.transform(pB)

##################################################################
########################### PGO TOOLS ############################

## Calculates a list of indexed pose pairs \f$((p_A, T_A), (p_B, T_B))\f$ for which robot \f$A\f$ and robot \f$B\f$ are viewing the same thing.
#  @param poses_A List of (int, SE3) indexed poses for robot \f$A\f$.
#  @param caminfo_A Tuple of camera info parameters for robot \f$A\f$: \f$(w,h,f_x,f_y)\f$ in pixels OR \f$(\text{FOV}_x, \text{FOV}_y)\f$ in radians.
#  @param max_depth_A Maximum depth at which robot \f$A\f$ can be expected to resolve objects for data association.
#  @param poses_B List of (int, SE3) indexed poses for robot \f$B\f$.
#  @param caminfo_B Tuple of camera info parameters for robot \f$B\f$: \f$(w,h,f_x,f_y)\f$ in pixels OR \f$(\text{FOV}_x, \text{FOV}_y)\f$ in radians.
#  @param max_depth_B Maximum depth at which robot \f$B\f$ can be expected to resolve objects for data association.
#  @param min_intra_idx_diff Minimum index difference (for the same robot) between subsequent overlapping views. Helps remove dense clusters.
#  @param min_inter_idx_diff Minimum index difference (between the two robots) between any overlapping views. Useful if both robot paths are identical and belonging to the same robot.
def getCommonCameraViewPoseIndices(poses_A, caminfo_A, max_depth_A, poses_B, caminfo_B, max_depth_B, min_intra_idx_diff=0, min_inter_idx_diff=0):
    if len(caminfo_A) == 4:
        caminfo_A = focal2fov(*caminfo_A)
    if len(caminfo_B) == 4:
        caminfo_B = focal2fov(*caminfo_B)

    def okay_to_check(i, j, ivals, jvals):
        i_closest_i = (min(ivals, key=lambda x:abs(x-i)) if len(ivals) > 0 else -1e10)
        j_closest_j = (min(jvals, key=lambda x:abs(x-j)) if len(jvals) > 0 else -1e10)
        return abs(i - i_closest_i) >= min_intra_idx_diff and abs(j - j_closest_j) >= min_intra_idx_diff and abs(i - j) > min_inter_idx_diff

    search_normals = [np.array([[1., 0., 0.]]).T, np.array([[0., 1., 0.]]).T, np.array([[0., 0., 1.]]).T]

    def cast_shadow_as_poly(points_mat, n):
        points_mat_2d = squashTo2DPlane(points_mat, n)
        points = MultiPoint([(points_mat_2d[0,i], points_mat_2d[1,i]) for i in range(points_mat_2d.shape[1])])
        return points.convex_hull

    A_indices = list()
    B_indices = list()

    indexed_pose_pairs = list()

    for i, pose_A in poses_A:
        A_pts = cam3DPolyPoints(pose_A, caminfo_A[0], caminfo_A[1], max_depth_A)
        for j, pose_B in poses_B:
            if okay_to_check(i, j, A_indices, B_indices):
                B_pts = cam3DPolyPoints(pose_B, caminfo_B[0], caminfo_B[1], max_depth_B)
                overlapping = True

                for n in search_normals:
                    A_shadow_poly = cast_shadow_as_poly(A_pts, n)
                    B_shadow_poly = cast_shadow_as_poly(B_pts, n)
                    if not (A_shadow_poly.overlaps(B_shadow_poly) or A_shadow_poly.within(B_shadow_poly) or B_shadow_poly.within(A_shadow_poly)):
                        overlapping = False
                        break
                
                if overlapping:
                    A_indices.append(i)
                    B_indices.append(j)
                    indexed_pose_pairs.append(((i, pose_A), (j, pose_B)))

    return indexed_pose_pairs

##################################################################
########################## PGO DATASETS ##########################

class SingleAgentPGODataset(object):
    def __init__(self, cam_info=None, true_poses=None, between_factors=None, loop_factors=None, alt_factors=None):
        self.cam_info = cam_info # (w, h, fx, fy)
        self.true_poses = (true_poses if not true_poses is None else list())                # [(i, SE3)]
        self.between_factors = (between_factors if not between_factors is None else list()) # [(i, j, SE3, 6x1 cov)]
        self.loop_factors = (loop_factors if not loop_factors is None else list())          # [(i, j, SE3, 6x1 cov)]
        self.alt_factors = (alt_factors if not alt_factors is None else list())             # [(i, h, cov)]

    #  @param loop_pose_cov \f$6\times 1\f$ Numpy array of loop closure covariances in translation and rotation.
    def synthesizeLoopFactors(self, loop_pose_cov=np.zeros((6,1)), max_cam_depth=5.0, min_idx_diff=100, plot=False):
        lc_stdev = np.linalg.cholesky(loop_pose_cov * np.eye(6))
        lc_index_pairs = getCommonCameraViewPoseIndices(self.true_poses, self.cam_info, max_cam_depth,
                                                        self.true_poses, self.cam_info, max_cam_depth,
                                                        min_intra_idx_diff=min_idx_diff,
                                                        min_inter_idx_diff=min_idx_diff)
        for lc_idx_pair in lc_index_pairs:
            pi = lc_idx_pair[0][0]
            Ti = lc_idx_pair[0][1]
            pj = lc_idx_pair[1][0]
            Tj = lc_idx_pair[1][1]
            lc_meas = SE3.Exp((Tj - Ti) + np.dot(lc_stdev, np.random.normal(np.zeros((6,1)), np.ones((6,1)), (6,1))))
            self.loop_factors.append((pi, pj, lc_meas, loop_pose_cov))

        if plot:
            path_x = [tp[1].t[0,0] for tp in self.true_poses]
            path_y = [tp[1].t[1,0] for tp in self.true_poses]
            _, ax = plt.subplots()
            ax.plot(path_x, path_y)
            for lf in self.loop_factors:
                lc_x = [self.true_poses[lf[0]][1].t[0,0], self.true_poses[lf[1]][1].t[0,0]]
                lc_y = [self.true_poses[lf[0]][1].t[1,0], self.true_poses[lf[1]][1].t[1,0]]
                ax.plot(lc_x, lc_y, 'r')
            plt.show()

    def synthesizeAltFactors(self, alt_cov=0.0):
        alt_stdev = sqrt(alt_cov)
        for pi, Ti in self.true_poses:
            alt_meas = Ti.t[2,0] + np.random.normal(0.0, alt_stdev)
            self.alt_factors.append((pi, alt_meas, alt_cov))

    def read(self, csvfile):
        csvf = open(csvfile, newline='')
        mode = ''
        reader = csv.reader(csvf, delimiter=',')
        for row in reader:
            if row[0] == '>':
                mode = row[1]
                continue
            else:
                if mode == 'CAMINFO':
                    self.cam_info = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                elif mode == 'TRUEPOSES':
                    self.true_poses.append((int(row[0]), 
                                            SE3(tx=float(row[1]), 
                                                ty=float(row[2]), 
                                                tz=float(row[3]), 
                                                qw=float(row[4]), 
                                                qx=float(row[5]), 
                                                qy=float(row[6]), 
                                                qz=float(row[7]))))
                elif mode == 'BETWEENFACTORS':
                    self.between_factors.append((int(row[0]),
                                                 int(row[1]),
                                                 SE3(tx=float(row[2]), 
                                                     ty=float(row[3]), 
                                                     tz=float(row[4]), 
                                                     qw=float(row[5]), 
                                                     qx=float(row[6]), 
                                                     qy=float(row[7]), 
                                                     qz=float(row[8])),
                                                 np.array([[float(row[9])],
                                                           [float(row[10])],
                                                           [float(row[11])],
                                                           [float(row[12])],
                                                           [float(row[13])],
                                                           [float(row[14])]])))
                elif mode == 'LOOPFACTORS':
                    self.loop_factors.append((int(row[0]),
                                              int(row[1]),
                                              SE3(tx=float(row[2]), 
                                                  ty=float(row[3]), 
                                                  tz=float(row[4]), 
                                                  qw=float(row[5]), 
                                                  qx=float(row[6]), 
                                                  qy=float(row[7]), 
                                                  qz=float(row[8])),
                                              np.array([[float(row[9])],
                                                        [float(row[10])],
                                                        [float(row[11])],
                                                        [float(row[12])],
                                                        [float(row[13])],
                                                        [float(row[14])]])))
                elif mode == 'ALTFACTORS':
                    self.alt_factors.append((int(row[0]), float(row[1]), float(row[2])))
                else:
                    print('WARNING: unrecognized mode for CSV read in SingleAgentPGODataset!')
        csvf.close()

    def write(self, csvfile):
        csvf = open(csvfile, 'w', newline='')
        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(['>', 'CAMINFO'])
        writer.writerow(list(self.cam_info))
        writer.writerow(['>', 'TRUEPOSES'])
        for true_pose in self.true_poses:
            writer.writerow([true_pose[0]] + list((true_pose[1].tq().T)[0]))
        writer.writerow(['>', 'BETWEENFACTORS'])
        for between_factor in self.between_factors:
            writer.writerow([between_factor[0], between_factor[1]] + list((between_factor[2].tq().T)[0]) + list((between_factor[3].T)[0]))
        writer.writerow(['>', 'LOOPFACTORS'])
        for loop_factor in self.loop_factors:
            writer.writerow([loop_factor[0], loop_factor[1]] + list((loop_factor[2].tq().T)[0]) + list((loop_factor[3].T)[0]))
        writer.writerow(['>', 'ALTFACTORS'])
        for alt_factor in self.alt_factors:
            writer.writerow(list(alt_factor))
        csvf.close()

class MultiAgentPGODataset(object):
    def __init__(self, single_agent_datasets=None, inter_loop_factors=None, range_factors=None):
        self.num_robots = 0
        self.cam_infos = list()
        self.true_poses = list()         # [(r, p, SE3)]
        self.between_factors = list()    # [(r, pi, pj, SE3, 6x1 cov)]
        self.intra_loop_factors = list() # [(r, pi, pj, SE3, 6x1 cov)]
        self.inter_loop_factors = list() # [(ri, rj, pi, pj, SE3, 6x1 cov)]
        self.range_factors = list()      # [(ri, rj, pi, pj, d, cov)]
        self.alt_factors = list()        # [(r, p, h, cov)]
        
        if not single_agent_datasets is None:
            r = 0
            for single_agent_dataset in single_agent_datasets:
                self.cam_infos.append(single_agent_dataset.cam_info)
                self.true_poses.extend([(r, tp[0], tp[1]) for tp in single_agent_dataset.true_poses])
                self.between_factors.extend([(r, bf[0], bf[1], bf[2], bf[3]) for bf in single_agent_dataset.between_factors])
                self.intra_loop_factors.extend([(r, lf[0], lf[1], lf[2], lf[3]) for lf in single_agent_dataset.loop_factors])
                self.alt_factors.extend([(r, af[0], af[1], af[2]) for af in single_agent_dataset.alt_factors])
                r += 1
            self.num_robots = r

        if not inter_loop_factors is None:
            self.inter_loop_factors = inter_loop_factors

        if not range_factors is None:
            self.range_factors = range_factors

    def _all_robot_combos(self):
        combos = list()
        for i in range(self.num_robots):
            for j in range(i):
                combos.append((i, j))
        return combos

    def synthesizeInterLoopFactors(self, loop_pose_cov=np.zeros((6,1)), max_cam_depth=5.0, min_idx_diff=100, plot=False):
        lc_stdev = np.linalg.cholesky(loop_pose_cov * np.eye(6))
        for robocombo in self._all_robot_combos():
            ri = robocombo[0]
            rj = robocombo[1]
            ri_poses = [(tp[1], tp[2]) for tp in self.true_poses if tp[0] == ri]
            rj_poses = [(tp[1], tp[2]) for tp in self.true_poses if tp[0] == rj]
            lc_index_pairs = getCommonCameraViewPoseIndices(ri_poses, self.cam_infos[ri], max_cam_depth,
                                                            rj_poses, self.cam_infos[rj], max_cam_depth,
                                                            min_intra_idx_diff=min_idx_diff)
            for lc_idx_pair in lc_index_pairs:
                pi = lc_idx_pair[0][0]
                Ti = lc_idx_pair[0][1]
                pj = lc_idx_pair[1][0]
                Tj = lc_idx_pair[1][1]
                lc_meas = SE3.Exp((Tj - Ti) + np.dot(lc_stdev, np.random.normal(np.zeros((6,1)), np.ones((6,1)), (6,1))))
                self.inter_loop_factors.append((ri, rj, pi, pj, lc_meas, loop_pose_cov))

            if plot:
                ri_path_x = [tp[1].t[0,0] for tp in ri_poses]
                ri_path_y = [tp[1].t[1,0] for tp in ri_poses]
                rj_path_x = [tp[1].t[0,0] for tp in rj_poses]
                rj_path_y = [tp[1].t[1,0] for tp in rj_poses]
                _, ax = plt.subplots()
                ax.plot(ri_path_x, ri_path_y, 'b')
                ax.plot(rj_path_x, rj_path_y, 'g')
                for lf in self.inter_loop_factors:
                    lc_x = [ri_poses[lf[2]][1].t[0,0], rj_poses[lf[3]][1].t[0,0]]
                    lc_y = [ri_poses[lf[2]][1].t[1,0], rj_poses[lf[3]][1].t[1,0]]
                    ax.plot(lc_x, lc_y, 'r')
                plt.show()

    def synthesizeRangeFactors(self, range_cov=0.0, every_n_poses=10):
        range_stdev = sqrt(range_cov)
        for robocombo in self._all_robot_combos():
            ri = robocombo[0]
            rj = robocombo[1]
            ri_poses = [(tp[1], tp[2]) for i, tp in enumerate(self.true_poses) if tp[0] == ri]
            rj_poses = [(tp[1], tp[2]) for i, tp in enumerate(self.true_poses) if tp[0] == rj]
            for pi, Ti in ri_poses:
                if pi % every_n_poses == 0:
                    for pj, Tj in rj_poses:
                        if pj == pi:
                            range_meas = np.linalg.norm(Tj.t - Ti.t) + np.random.normal(0.0, range_stdev)
                            self.range_factors.append((ri, rj, pi, pj, range_meas, range_cov))
                            break

    def read(self, csvfile):
        csvf = open(csvfile, newline='')
        mode = ''
        reader = csv.reader(csvf, delimiter=',')
        for row in reader:
            if row[0] == '>':
                mode = row[1]
                continue
            else:
                if mode == 'CAMINFO':
                    self.cam_infos.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
                elif mode == 'TRUEPOSES':
                    self.true_poses.append((int(row[0]),
                                            int(row[1]), 
                                            SE3(tx=float(row[2]), 
                                                ty=float(row[3]), 
                                                tz=float(row[4]), 
                                                qw=float(row[5]), 
                                                qx=float(row[6]), 
                                                qy=float(row[7]), 
                                                qz=float(row[8]))))
                elif mode == 'BETWEENFACTORS':
                    self.between_factors.append((int(row[0]),
                                                 int(row[1]),
                                                 int(row[2]),
                                                 SE3(tx=float(row[3]), 
                                                     ty=float(row[4]), 
                                                     tz=float(row[5]), 
                                                     qw=float(row[6]), 
                                                     qx=float(row[7]), 
                                                     qy=float(row[8]), 
                                                     qz=float(row[9])),
                                                 np.array([[float(row[10])],
                                                           [float(row[11])],
                                                           [float(row[12])],
                                                           [float(row[13])],
                                                           [float(row[14])],
                                                           [float(row[15])]])))
                elif mode == 'INTRALOOPFACTORS':
                    self.intra_loop_factors.append((int(row[0]),
                                                    int(row[1]),
                                                    int(row[2]),
                                                    SE3(tx=float(row[3]), 
                                                        ty=float(row[4]), 
                                                        tz=float(row[5]), 
                                                        qw=float(row[6]), 
                                                        qx=float(row[7]), 
                                                        qy=float(row[8]), 
                                                        qz=float(row[9])),
                                                    np.array([[float(row[10])],
                                                                [float(row[11])],
                                                                [float(row[12])],
                                                                [float(row[13])],
                                                                [float(row[14])],
                                                                [float(row[15])]])))
                elif mode == 'ALTFACTORS':
                    self.alt_factors.append((int(row[0]), int(row[1]), float(row[2]), float(row[3])))
                elif mode == 'INTERLOOPFACTORS':
                    self.inter_loop_factors.append((int(row[0]),
                                                    int(row[1]),
                                                    int(row[2]),
                                                    int(row[3]),
                                                    SE3(tx=float(row[4]), 
                                                        ty=float(row[5]), 
                                                        tz=float(row[6]), 
                                                        qw=float(row[7]), 
                                                        qx=float(row[8]), 
                                                        qy=float(row[9]), 
                                                        qz=float(row[10])),
                                                    np.array([[float(row[11])],
                                                                [float(row[12])],
                                                                [float(row[13])],
                                                                [float(row[14])],
                                                                [float(row[15])],
                                                                [float(row[16])]])))
                elif mode == 'RANGEFACTORS':
                    self.range_factors.append((int(row[0]),
                                               int(row[1]),
                                               int(row[2]),
                                               int(row[3]),
                                               float(row[4]),
                                               float(row[5])))
                else:
                    print('WARNING: unrecognized mode for CSV read in MultiAgentPGODataset!')
        self.num_robots = len(self.cam_infos)
        csvf.close()

    def write(self, csvfile):
        csvf = open(csvfile, 'w', newline='')
        writer = csv.writer(csvf, delimiter=',')
        writer.writerow(['>', 'CAMINFO'])
        for cam_info in self.cam_infos:
            writer.writerow(list(cam_info))
        writer.writerow(['>', 'TRUEPOSES'])
        for true_pose in self.true_poses:
            writer.writerow([true_pose[0], true_pose[1]] + list((true_pose[2].tq().T)[0]))
        writer.writerow(['>', 'BETWEENFACTORS'])
        for between_factor in self.between_factors:
            writer.writerow([between_factor[0], between_factor[1], between_factor[2]] + list((between_factor[3].tq().T)[0]) + list((between_factor[4].T)[0]))
        writer.writerow(['>', 'INTRALOOPFACTORS'])
        for loop_factor in self.intra_loop_factors:
            writer.writerow([loop_factor[0], loop_factor[1], loop_factor[2]] + list((loop_factor[3].tq().T)[0]) + list((loop_factor[4].T)[0]))
        writer.writerow(['>', 'ALTFACTORS'])
        for alt_factor in self.alt_factors:
            writer.writerow(list(alt_factor))
        writer.writerow(['>', 'INTERLOOPFACTORS'])
        for loop_factor in self.inter_loop_factors:
            writer.writerow([loop_factor[0], loop_factor[1], loop_factor[2], loop_factor[3]] + list((loop_factor[4].tq().T)[0]) + list((loop_factor[5].T)[0]))
        writer.writerow(['>', 'RANGEFACTORS'])
        for range_factor in self.range_factors:
            writer.writerow(list(range_factor))
        csvf.close()

##################################################################
########################## ROSBAG PARSE ##########################

class ROSbagParserBase(object):
    def __init__(self, bagfile):
        if not _ROS_bag:
            raise ROSNotFoundError('ROSbag parser not found on system.')
        self.bagfile = bagfile
        self.bag = rosbag.Bag(bagfile)
        self.t0 = None
        self.tf = None
        for _, _, t in self.bag.read_messages():
            if self.t0 == None or t < self.t0:
                self.t0 = t
            if self.tf == None or t > self.tf:
                self.tf = t
        self.t0 = self.t0.to_sec()
        self.tf = self.tf.to_sec()
        # print('t: (%f, %f)' % (self.t0, self.tf))

    def process_msgs_by_type(self, conv_f, msg_type, topic_name=None):
        print('Polling {} for messages of type {}{}'.format(self.bagfile, msg_type,
            '...' if topic_name == None else ' and topic name %s...' % topic_name))
        data = None
        for topic, msg, t in self.bag.read_messages():
            if msg._type == msg_type and ((not topic_name is None and topic == topic_name) or topic_name is None):
                datum = np.vstack((np.array([[t.to_sec()-self.t0]]), conv_f(msg)))
                data = (datum if data is None else np.hstack((data, datum)))
        if not data is None:
            n = data.shape[1]
            dt = self.tf-self.t0
            print('  Collected {} messages over {} s (>= {} Hz).'.format(n, dt, n/dt))
        return data

    def close(self):
        self.bag.close()

class ROSbagCameraDataset(ROSbagParserBase):
    def __init__(self, bagfile, img_topic_name=None, info_topic_name=None):
        if not _ROS_bridge:
            raise ROSNotFoundError('Cannot create ROSbag camera dataset without cv_bridge.')
        super(ROSbagCameraDataset, self).__init__(bagfile)
        self.img_topic = img_topic_name
        self.info_topic = info_topic_name
        
        self.t_data = None
        self.img_data = list()
        self.ros_encoding = None
        self.h = None
        self.w = None
        self.dist_model = None
        self.D = None
        self.K = None
        self.R = None
        self.P = None

        self.bridge = CvBridge()

        print('Polling {} for Image and CameraInfo messages...'.format(bagfile))
        img_type = Image()._type
        info_type = CameraInfo()._type
        for topic, msg, t in self.bag.read_messages():
            if msg._type == img_type and ((not self.img_topic is None and topic == self.img_topic) or self.img_topic is None):
                t_datum = np.array([[t.to_sec()-self.t0]])
                self.t_data = (t_datum if self.t_data is None else np.hstack((self.t_data, t_datum)))
                self.ros_encoding = msg.encoding
                self.img_data.append(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
            elif msg._type == info_type and ((not self.info_topic is None and topic == self.info_topic) or self.info_topic is None) and self.h is None:
                self.h = msg.height
                self.w = msg.width
                self.dist_model = msg.distortion_model
                self.D = np.array(msg.D)
                self.K = np.array(msg.K).reshape((3,3))
                self.R = np.array(msg.R).reshape((3,3))
                self.P = np.array(msg.P).reshape((3,4))

        if not self.img_data is None:
            print('  Collected {} images.{}'.format(len(self.img_data), '\n  Collected camera info data.' if self.h != None else ''))

        self.close()

    def setCameraInfo(self, h, w, K, dist_model='plumb_bob', D=None, R=None, P=None):
        self.h = h
        self.w = w
        self.K = K
        self.dist_model = dist_model
        self.D = D
        self.R = R
        self.P = P

    def write(self, bagfile, img_topic_name='/camera/raw', info_topic_name='/camera/info', spec='w'):
        print('Writing {}{} -> {}.'.format(img_topic_name, '' if self.h is None else ', %s' % info_topic_name, bagfile))
        with rosbag.Bag(bagfile, spec) as bag:
            for i in range(self.t_data.size):
                t = rospy.Time(self.t_data[0,i])
                cam_msg = self.bridge.cv2_to_imgmsg(self.img_data[i], encoding=self.ros_encoding)
                cam_msg.header.stamp = t
                bag.write(img_topic_name, cam_msg, t)
                if not self.h is None:
                    info_msg = CameraInfo()
                    info_msg.height = self.h
                    info_msg.width = self.w
                    info_msg.K = list(self.K.reshape((1,9))[0])
                    if not self.dist_model is None:
                        info_msg.distortion_model = self.dist_model
                    if not self.D is None:
                        info_msg.D = list(self.D)
                    if not self.R is None:
                        info_msg.R = list(self.R.reshape((1,9))[0])
                    if not self.P is None:
                        info_msg.P = list(self.P.reshape((1,12))[0])
                    info_msg.header.stamp = t
                    bag.write(info_topic_name, info_msg, t)

class ROSbagImuDataset(ROSbagParserBase):
    def __init__(self, bagfile, topic_name=None):
        super(ROSbagImuDataset, self).__init__(bagfile)
        self.topic = topic_name

        self.t_data = None
        self.acc_data = None
        self.gyr_data = None

        data = self.process_msgs_by_type(self.convImu, Imu()._type, topic_name)
        if not data is None:
            self.t_data = data[0:1,:]
            self.acc_data = data[1:4,:]
            self.gyr_data = data[4:7,:]

        self.close()

    def write(self, bagfile, topic_name='/imu', spec='w'):
        print('Writing {} -> {}.'.format(topic_name, bagfile))
        with rosbag.Bag(bagfile, spec) as bag:
            for i in range(self.t_data.shape[1]):
                t = rospy.Time(self.t_data[0,i])
                msg = Imu()
                msg.linear_acceleration.x = self.acc_data[0,i]
                msg.linear_acceleration.y = self.acc_data[1,i]
                msg.linear_acceleration.z = self.acc_data[2,i]
                msg.angular_velocity.x = self.gyr_data[0,i]
                msg.angular_velocity.y = self.gyr_data[1,i]
                msg.angular_velocity.z = self.gyr_data[2,i]
                msg.header.stamp = t
                bag.write(topic_name, msg, t)

    def convImu(self, msg):
        return np.array([[msg.linear_acceleration.x],
                         [msg.linear_acceleration.y],
                         [msg.linear_acceleration.z],
                         [msg.angular_velocity.x],
                         [msg.angular_velocity.y],
                         [msg.angular_velocity.z]])

class ROSbagStateDataset(ROSbagParserBase):
    def __init__(self, bagfile, topic_name=None):
        super(ROSbagStateDataset, self).__init__(bagfile)
        self.topic = topic_name
        
        self.t_data = None
        self.pos_data = None
        self.att_data = None
        self.vel_data = None
        self.omg_data = None

        data = self.process_msgs_by_type(self.convOdometry, Odometry()._type, topic_name)
        if not data is None:
            self.t_data = data[0:1,:]
            self.pos_data = data[1:4,:]
            self.att_data = data[4:8,:]
            self.vel_data = data[8:11,:]
            self.omg_data = data[11:14,:]
            self.close()
            return

        data = self.process_msgs_by_type(self.convPose, Pose()._type, topic_name)
        data = (self.process_msgs_by_type(self.convPoseStamped, PoseStamped()._type, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convPoseWithCovariance, PoseWithCovariance()._type, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convPoseWithCovarianceStamped, PoseWithCovarianceStamped()._type, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convTransform, Transform()._type, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convTransformStamped, TransformStamped()._type, topic_name) if data is None else data)
        if not data is None:
            self.t_data = data[0:1,:]
            self.pos_data = data[1:4,:]
            self.att_data = data[4:8,:]
            self.close()
            return

        data = self.process_msgs_by_type(self.convPoint, Point()._type, topic_name)
        data = (self.process_msgs_by_type(self.convPointStamped, PointStamped()._type, topic_name) if data is None else data)
        if not data is None:
            self.t_data = data[0:1,:]
            self.pos_data = data[1:4,:]
            
        self.close()

    def setStateValues(self, t, pos=None, att=None, vel=None, omg=None):
        n = len(t)
        self.t_data = np.zeros((1,n))
        for i in range(n):
            self.t_data[0,i] = t[i]
        if not pos is None:
            self.pos_data = np.zeros((3,n))
            for i in range(n):
                self.pos_data[:,i:i+1] = pos[i]
        if not att is None:
            self.att_data = np.zeros((4,n))
            for i in range(n):
                self.att_data[:,i:i+1] = att[i]
        if not vel is None:
            self.vel_data = np.zeros((3,n))
            for i in range(n):
                self.vel_data[:,i:i+1] = vel[i]
        if not omg is None:
            self.omg_data = np.zeros((3,n))
            for i in range(n):
                self.omg_data[:,i:i+1] = omg[i]
        # FIXME interpolate

    def write(self, bagfile, topic_name='/state', spec='w'):
        print('Writing {} -> {}.'.format(topic_name, bagfile))
        with rosbag.Bag(bagfile, spec) as bag:
            for i in range(self.t_data.shape[1]):
                t = rospy.Time(self.t_data[0,i])
                msg = Odometry()
                if not self.pos_data is None:
                    msg.pose.pose.position.x = self.pos_data[0,i]
                    msg.pose.pose.position.y = self.pos_data[1,i]
                    msg.pose.pose.position.z = self.pos_data[2,i]
                if not self.att_data is None:
                    msg.pose.pose.orientation.w = self.att_data[0,i]
                    msg.pose.pose.orientation.x = self.att_data[1,i]
                    msg.pose.pose.orientation.y = self.att_data[2,i]
                    msg.pose.pose.orientation.z = self.att_data[3,i]
                if not self.vel_data is None:
                    msg.twist.twist.linear.x = self.vel_data[0,i]
                    msg.twist.twist.linear.y = self.vel_data[1,i]
                    msg.twist.twist.linear.z = self.vel_data[2,i]
                if not self.omg_data is None:
                    msg.twist.twist.angular.x = self.omg_data[0,i]
                    msg.twist.twist.angular.y = self.omg_data[1,i]
                    msg.twist.twist.angular.z = self.omg_data[2,i]
                msg.header.stamp = t
                bag.write(topic_name, msg, t)

    def convOdometry(self, msg):
        return np.array([[msg.pose.pose.position.x],
                         [msg.pose.pose.position.y],
                         [msg.pose.pose.position.z],
                         [msg.pose.pose.orientation.w],
                         [msg.pose.pose.orientation.x],
                         [msg.pose.pose.orientation.y],
                         [msg.pose.pose.orientation.z],
                         [msg.twist.twist.linear.x],
                         [msg.twist.twist.linear.y],
                         [msg.twist.twist.linear.z],
                         [msg.twist.twist.angular.x],
                         [msg.twist.twist.angular.y],
                         [msg.twist.twist.angular.z]])

    def convPose(self, msg):
        return np.array([[msg.position.x],
                         [msg.position.y],
                         [msg.position.z],
                         [msg.orientation.w],
                         [msg.orientation.x],
                         [msg.orientation.y],
                         [msg.orientation.z]])

    def convPoseStamped(self, msg):
        return self.convPose(msg.pose)

    def convPoseWithCovariance(self, msg):
        return self.convPose(msg.pose)

    def convPoseWithCovarianceStamped(self, msg):
        return self.convPose(msg.pose.pose)

    def convTransform(self, msg):
        return np.array([[msg.translation.x],
                         [msg.translation.y],
                         [msg.translation.z],
                         [msg.rotation.w],
                         [msg.rotation.x],
                         [msg.rotation.y],
                         [msg.rotation.z]])

    def convTransformStamped(self, msg):
        return self.convTransform(msg.transform)

    def convPoint(self, msg):
        return np.array([[msg.x],
                         [msg.y],
                         [msg.z]])

    def convPointStamped(self, msg):
        return self.convPoint(msg.point)

## Dataset container for single-agent, visual-inertial, monocular SLAM.
class ROSbagSingleAgentViMonoSlamDataset(object):
    ## Instantiates the dataset from a ROSbag file.
    #  @param bagfile The file path of the input ROSbag file.
    #  @param truth_topic (Optional) Topic to take truth values from. If set to None, will look for topics with message type Odometry, Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped, Transform, TransformStamped, Point, or PointStamped.
    #  @param image_topic (Optional) Topic to take images from. If set to None, will look for topics with message type Image.
    #  @param imu_topic (Optional) Topic to take IMU measurements from. If set to None, will look for topics with message type Imu.
    #  @param cam_info_topic (Optional) Topic to take camera info from. If set to None, will look for topics with message type CameraInfo. 
    #  @param vio_topic (Optional) Topic to take VIO measurements from. If set to None, will look for topics with same message type as truth.
    def __init__(self, bagfile, truth_topic='/truth', image_topic='/image', imu_topic='/imu', cam_info_topic='/caminfo', vio_topic='/vio'):
        self.truth  = ROSbagStateDataset(bagfile, truth_topic)
        self.camera = ROSbagCameraDataset(bagfile, image_topic, cam_info_topic)
        self.imu    = ROSbagImuDataset(bagfile, imu_topic)
        self.vio    = ROSbagStateDataset(bagfile, vio_topic)

    ## Set the monocular camera intrinsics.
    #  @param w Image width.
    #  @param h Image height.
    #  @param fu Focal length in x.
    #  @param fv Focal length in y.
    #  @param cu Center pixel in x.
    #  @param cv Center pixel in y.
    #  @param D List of radial-tangential distortion parameters (k1, k2, t1, t2, k3)
    def setCameraIntrinsics(self, w, h, fu, fv, cu, cv, D=(0.,0.,0.,0.,0.)):
        K = np.array([[fu, 0, cu],
                      [0, fv, cv],
                      [0, 0, 1.0]])
        self.camera.setCameraInfo(h, w, K, 'plumb_bob', np.array(D))

    ## Set the truth state fields. All input lists should be synchronized and of the same length, else set to None.
    #  @param t List of time values.
    #  @param pos List of \f$3\times 1\f$ numpy array world-frame position vectors \f$p_{B/W}^W\f$.
    #  @param att List of \f$4\times 1\f$ numpy array quaternion vectors representing \f$R_B^W\f$ and of the form \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
    #  @param vel List of \f$3\times 1\f$ numpy array body-frame velocity vectors \f$v_{B/W}^B\f$.
    #  @param omg List of \f$3\times 1\f$ numpy array body-frame angular velocity vectors \f$\omega_{B/W}^B\f$.
    def setTruthFields(self, t, pos, att, vel, omg):
        self.truth.setStateValues(t, pos, att, vel, omg)

    ## Applies a rigid body transform to all available truth state fields, resulting in a transformed dataset.
    #  @param T_U_B The transform SE3 object, \f$T_U^B\f$. 
    #  @param T_N_W The transform SE3 object, \f$T_N^W\f$.
    #
    #  Assumes that you are trying to shift the truth measurements from the vehicle body's center of
    #  mass (\f$B\f$) to another frame still rigidly attached to the vehicle body (\f$U\f$). The second argument
    #  allows you to simultaneously modify the world frame from \f$W\rightarrow N\f$, which again assumes that 
    #  frame \f$N\f$ is rigidly attached to frame \f$W\f$. 
    def transformTruthFrame(self, T_U_B, T_N_W=SE3.identity()):
        q_U_B = T_U_B.q
        t_UB_B = T_U_B.t
        q_N_W = T_N_W.q
        t_NW_W = T_N_W.t
        print('Transforming dataset truth measurements:\nR_U^B:\n{}\nt_U/B:\n{}\nR_N^W:\n{}\nt_N/W:\n{}\n'.format(q_U_B.R(), t_UB_B, q_N_W.R(), t_NW_W))

        for i in range(self.truth.t_data.shape[1]):
            p_BW_W = np.array(self.truth.pos_data[:,i:i+1], copy=True)
            q_B_W = SO3(np.array(self.truth.att_data[:,i:i+1], copy=True))

            self.truth.pos_data[:,i:i+1] = np.dot(q_N_W.inverse().R(), p_BW_W - t_NW_W + np.dot(q_B_W.R(), t_UB_B))
            self.truth.att_data[:,i:i+1] = (q_N_W.inverse() * q_B_W * q_U_B).arr

            if (not self.truth.vel_data is None) and (not self.truth.omg_data is None):
                v_BW_B = np.array(self.truth.vel_data[:,i:i+1], copy=True)
                w_BW_B = np.array(self.truth.omg_data[:,i:i+1], copy=True)
                self.truth.vel_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), v_BW_B + np.cross(w_BW_B, t_UB_B, axis=0))
                self.truth.omg_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), w_BW_B)

    ## Applies a rigid body transform to all available VIO fields, resulting in a transformed dataset.
    #  @param T_U_B The transform SE3 object, \f$T_U^B\f$. 
    #  @param T_N_W The transform SE3 object, \f$T_N^W\f$.
    #  @param t_off Desired time offset to match VIO timing with truth.
    #
    #  Assumes that you are trying to shift the VIO measurements from the vehicle body's center of
    #  mass (\f$B\f$) to another frame still rigidly attached to the vehicle body (\f$U\f$). The second argument
    #  allows you to simultaneously modify the world frame from \f$W\rightarrow N\f$, which again assumes that 
    #  frame \f$N\f$ is rigidly attached to frame \f$W\f$. 
    def transformVIOFrame(self, T_U_B, T_N_W=SE3.identity(), t_off=0.0):
        q_U_B = T_U_B.q
        t_UB_B = T_U_B.t
        q_N_W = T_N_W.q
        t_NW_W = T_N_W.t
        print('Transforming dataset VIO measurements:\nR_U^B:\n{}\nt_U/B:\n{}\nR_N^W:\n{}\nt_N/W:\n{}\nt_off:{}\n'.format(q_U_B.R(), t_UB_B, q_N_W.R(), t_NW_W, t_off))

        self.vio.t_data += t_off

        for i in range(self.vio.t_data.shape[1]):
            p_BW_W = np.array(self.vio.pos_data[:,i:i+1], copy=True)
            q_B_W = SO3(np.array(self.vio.att_data[:,i:i+1], copy=True))

            self.vio.pos_data[:,i:i+1] = np.dot(q_N_W.inverse().R(), p_BW_W - t_NW_W + np.dot(q_B_W.R(), t_UB_B))
            self.vio.att_data[:,i:i+1] = (q_N_W.inverse() * q_B_W * q_U_B).arr

            if (not self.vio.vel_data is None) and (not self.vio.omg_data is None):
                v_BW_B = np.array(self.vio.vel_data[:,i:i+1], copy=True)
                w_BW_B = np.array(self.vio.omg_data[:,i:i+1], copy=True)
                self.vio.vel_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), v_BW_B + np.cross(w_BW_B, t_UB_B, axis=0))
                self.vio.omg_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), w_BW_B)

    ## Transforms the velocity truth fields from the world frame to the body frame using the attitude truth fields.
    #
    #  Assumes that the attitude truth fields represent \f$R_B^W\f$ and are of the form \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
    def rotateVelocityTruthToBody(self):
        print('Rotating truth velocities into the body frame.')
        for i in range(self.truth.t_data.shape[1]):
            q_B_W = SO3(np.array(self.truth.att_data[:,i:i+1], copy=True))
            v_BW_W = np.array(self.truth.vel_data[:,i:i+1], copy=True)

            self.truth.vel_data[:,i:i+1] = np.dot(q_B_W.inverse().R(), v_BW_W)

    ## Transforms the VIO velocity fields from the world frame to the body frame using the VIO attitude fields.
    #
    #  Assumes that the VIO attitude fields represent \f$R_B^W\f$ and are of the form \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
    def rotateVIOVelocityToBody(self):
        print('Rotating VIO velocities into the body frame.')
        for i in range(self.vio.t_data.shape[1]):
            q_B_W = SO3(np.array(self.vio.att_data[:,i:i+1], copy=True))
            v_BW_W = np.array(self.vio.vel_data[:,i:i+1], copy=True)

            self.vio.vel_data[:,i:i+1] = np.dot(q_B_W.inverse().R(), v_BW_W)

    ## Applies a rigid body transform to all IMU measurements, resulting in a transformed dataset.
    #  @param T The transform SE3 object, \f$T_U^B\f$. 
    #
    #  Assumes that you are trying to shift the IMU measurements from the vehicle body's center of
    #  mass (\f$B\f$) to another frame still rigidly attached to the vehicle body (\f$U\f$). 
    #  Implements the formulas derived at https://notes.andrewtorgesen.com/doku.php?id=public:imu#shifting_measurements_in_se_3.
    def transformImuFrame(self, T):
        # https://notes.andrewtorgesen.com/doku.php?id=public:imu
        q_U_B = T.q
        t_UB_B = T.t
        print('Transforming dataset IMU measurements:\nR:\n{}\nt:\n{}\n'.format(q_U_B.R(), t_UB_B))

        wd = Differentiator((3,1))
        t_prev = None
        for i in range(self.imu.t_data.shape[1]):
            t = self.imu.t_data[0,i]
            Ts = (t-t_prev if i > 0 else 0)
            t_prev = t
            a = np.array(self.imu.acc_data[:,i:i+1], copy=True)
            w = np.array(self.imu.gyr_data[:,i:i+1], copy=True)

            O = SO3.hat(wd.calculate(w, Ts)) + np.dot(w, w.T) - np.dot(w.T, w) * np.eye(3)
            self.imu.acc_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), a + np.dot(O, t_UB_B))
            self.imu.gyr_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), w)

    ## Numerically differentiates the truth pose fields to obtain body-frame translational and angular velocities.
    #
    #  Assumes that position is expressed in the world frame \f$p_{B/W}^W\f$ and that attitude represents \f$R_B^W\f$ and is of the form \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
    def differentiateTruthPoses(self):
        v_W = Differentiator((3,1))
        w_B = Differentiator((3,1))
        n = self.truth.t_data.shape[1]
        if self.truth.vel_data is None:
            self.truth.vel_data = np.zeros((3,n))
        if self.truth.omg_data is None:
            self.truth.omg_data = np.zeros((3,n))
        t_prev = None
        for i in range(n):
            t = self.truth.t_data[0,i]
            Ts = (t-t_prev if i > 0 else 0)
            t_prev = t
            p_BW_W = np.array(self.truth.pos_data[:,i:i+1], copy=True)
            q_B_W = SO3(np.array(self.truth.att_data[:,i:i+1], copy=True))
            self.truth.vel_data[:,i:i+1] = np.dot(q_B_W.inverse().R(), v_W.calculate(p_BW_W, Ts))
            self.truth.omg_data[:,i:i+1] = w_B.calculate(q_B_W, Ts)

    ## Generates fake IMU measurements using the dataset's true state fields.
    #  @param g The gravity vector, expressed in the world frame. A \f$3\times 1\f$ numpy array.
    #  @param sigma_eta A tuple of zero-mean noise standard deviations \f$(\sigma_{\eta,\text{accel}}, \sigma_{\eta,\text{gyro}})\f$.
    #  @param kappa A tuple of bounds for the initial bias value \f$(\kappa_{\beta,\text{accel}}, \kappa_{\beta,\text{gyro}})\f$.
    #  @param sigma_beta A tuple of random walk standard deviations \f$(\sigma_{\beta,\text{accel}}, \sigma_{\beta,\text{gyro}})\f$.
    #  
    #  The default parameters generate noise- and bias-less measurements, assuming that the world z-axis points up.
    #  Implements the formulas derived at https://notes.andrewtorgesen.com/doku.php?id=public:imu#synthesizing_measurements.
    def synthesizeImu(self, g=np.array([[0,0,-9.81]]).T, sigma_eta=(0,0), kappa=(0,0), sigma_beta=(0,0)):
        # g is 3x1 vector in W frame, default arg assumes that z_W points up
        # https://notes.andrewtorgesen.com/doku.php?id=public:imu

        if not self.truth.t_data is None:
            imu_t = [t for t in np.hsplit(self.truth.t_data, self.truth.t_data.shape[1])]
        else:
            raise Exception('Truth data required for IMU synthesis.')
        n = len(imu_t)

        if not self.truth.att_data is None:
            imu_q = [SO3(np.array(q, copy=True)) for q in np.hsplit(self.truth.att_data, n)]
            if not self.truth.omg_data is None:
                print('Synthesizing rate gyro data from body-frame angular velocity truth.')
                imu_w = [np.array(w, copy=True) for w in np.hsplit(self.truth.omg_data, n)]
            else:
                print('Synthesizing rate gyro data from attitude truth.')
                imu_w = list()
                w_B = Differentiator((3,1))
                t_prev = None
                for i in range(n):
                    t  = imu_t[i]
                    Ts = (t-t_prev if i > 0 else 0)
                    t_prev = t
                    imu_w.append(w_B.calculate(SO3(self.truth.att_data[:,i:i+1]), Ts))
        else:
            raise Exception('Some kind of attitude truth data required for IMU synthesis.')

        if not self.truth.vel_data is None:
            print('Syntheszing accel data from body-frame velocity truth.')
            imu_v_W = list()
            for i in range(n):
                imu_v_W.append(np.dot(imu_q[i].R(), np.array(self.truth.vel_data[:,i:i+1], copy=True)))
        elif not self.truth.pos_data is None:
            print('Synthesizing accel data from world-frame position truth.')
            imu_v_W = list()
            v_W = Differentiator((3,1))
            t_prev = None
            for i in range(n):
                t = imu_t[i]
                Ts = (t-t_prev if i > 0 else 0)
                t_prev = t
                imu_v_W.append(v_W.calculate(self.truth.pos_data[:,i:i+1], Ts))
        else:
            raise Exception('Some kind of translation truth data required for IMU synthesis.')

        beta_a = np.random.uniform(-kappa[0], kappa[0], size=(3,1))
        beta_w = np.random.uniform(-kappa[1], kappa[1], size=(3,1))

        self.imu.t_data = np.zeros((1, n))
        self.imu.acc_data = np.zeros((3, n))
        self.imu.gyr_data = np.zeros((3, n))
        a_I = Differentiator((3,1))
        t_prev = None
        for i, (t, q, w, v) in enumerate(zip(imu_t, imu_q, imu_w, imu_v_W)):
            Ts = (t-t_prev if i > 0 else 0)
            t_prev = t
            beta_a += np.random.normal(0, sigma_beta[0], size=(3,1)) * Ts
            beta_w += np.random.normal(0, sigma_beta[1], size=(3,1)) * Ts
            eta_a = np.random.normal(0, sigma_eta[0], size=(3,1))
            eta_w = np.random.normal(0, sigma_eta[1], size=(3,1))
            self.imu.t_data[:,i:i+1] = t
            self.imu.acc_data[:,i:i+1] = np.dot(q.inverse().R(), a_I.calculate(v, Ts) - g) + eta_a + beta_a
            self.imu.gyr_data[:,i:i+1] = w + eta_w + beta_w

    ## Constructs a SingleAgentPGODataset (no loop closures or altitude factors) object using truth, camera info, and VIO. Between factor covariance is calculated using truth pose data.
    #  @param plot Visualize VIO covariance calculation.
    #  Assumes that VIO and truth are aligned geometrically and temporally (e.g., via transformVIOFrame()).
    def getPGODataset(self, plot=False):
        cam_info = (self.camera.w, self.camera.h, self.camera.K[0,0], self.camera.K[1,1])
        truth_interp = LinearInterpolator([self.truth.t_data[0,i] for i in range(self.truth.t_data.shape[1])],
                                          [SE3(np.vstack((self.truth.pos_data[:,i:i+1], self.truth.att_data[:,i:i+1]))) for i in range(self.truth.t_data.shape[1])])
        n = self.vio.t_data.shape[1]
        true_poses = [(i, truth_interp.y(self.vio.t_data[0,i])) for i in range(n)]

        between_factors_sans_cov = list()
        between_factors = list()
        vio_errors = np.zeros((6, n-1))
        for i in range(n-1):
            j = i+1

            Ti_true = true_poses[i][1]
            Tj_true = true_poses[j][1]
            Tij_true = SE3.Exp(Tj_true - Ti_true)

            Ti = SE3(np.vstack((self.vio.pos_data[:,i:i+1], self.vio.att_data[:,i:i+1])))
            Tj = SE3(np.vstack((self.vio.pos_data[:,j:j+1], self.vio.att_data[:,j:j+1])))
            Tij = SE3.Exp(Tj - Ti)

            between_factors_sans_cov.append((i, j, Tij))

            vio_errors[:,i:i+1] = Tij - Tij_true

        pos_x_stdev = getZeroMeanGaussianErrorStdDev(vio_errors[0,:], covariance_gating=True, title='X-Position Error Stdev', plot=plot)
        pos_y_stdev = getZeroMeanGaussianErrorStdDev(vio_errors[1,:], covariance_gating=True, title='Y-Position Error Stdev', plot=plot)
        pos_z_stdev = getZeroMeanGaussianErrorStdDev(vio_errors[2,:], covariance_gating=True, title='Z-Position Error Stdev', plot=plot)
        rot_x_stdev = getZeroMeanGaussianErrorStdDev(vio_errors[3,:], covariance_gating=True, title='X-Attitude Error Stdev', plot=plot)
        rot_y_stdev = getZeroMeanGaussianErrorStdDev(vio_errors[4,:], covariance_gating=True, title='Y-Attitude Error Stdev', plot=plot)
        rot_z_stdev = getZeroMeanGaussianErrorStdDev(vio_errors[5,:], covariance_gating=True, title='Z-Attitude Error Stdev', plot=plot)

        for bfsc in between_factors_sans_cov:
            between_factors.append((bfsc[0], bfsc[1], bfsc[2], np.array([[pos_x_stdev**2], [pos_y_stdev**2], [pos_z_stdev**2],
                                                                         [rot_x_stdev**2], [rot_y_stdev**2], [rot_z_stdev**2]])))

        return SingleAgentPGODataset(cam_info, true_poses, between_factors)

    ## Write the current dataset contents to a new ROSbag file.
    #  @param bagfile The path to the new bagfile.
    #  @param truth_topic New topic name for the truth nav_msgs/Odometry field.
    #  @param image_topic New topic name for the sensor_msgs/Image field.
    #  @param imu_topic New topic name for the sensor_msgs/Imu field.
    #  @param cam_info_topic New topic name for the sensor_msgs/CameraInfo field.
    #  @param vio_topic New topic name for the VIO nav_msgs/Odometry field.
    def writeBag(self, bagfile, truth_topic='/truth', image_topic='/rgb', imu_topic='/imu', cam_info_topic='/rgb/info', vio_topic='/vio'):
        self.truth.write(bagfile, topic_name=truth_topic)
        self.camera.write(bagfile, img_topic_name=image_topic, info_topic_name=cam_info_topic, spec='a')
        self.imu.write(bagfile, topic_name=imu_topic, spec='a')
        self.vio.write(bagfile, topic_name=vio_topic, spec='a')

def test_SO3():
    R1 = SO3.random()
    print(R1)
    R2 = R1 + np.array([[0.5, 0.2, 0.1]]).T
    print(R2)
    print(R2-R1)

def test_SE3():
    np.random.seed(144440)
    Trand = SE3.random()
    Trond = SE3.random()
    dT = Trond-Trand
    T2 = Trand + dT

    print((Trand * Trond).tq())
    print()
    print(Trand.inverse().tq())
    print()
    print(dT)
    print()

    print(Trond.tq())
    print()
    print((SE3.Exp(SE3.Log(Trond))).tq())
    print()
    print(T2.tq())
    print()
    print(dT)
    print()
    print(SE3.Log(SE3.Exp(dT)))

def test_qR_convs():
    # np.random.seed(144440)
    uT = np.random.random((1,3))
    uT /= np.linalg.norm(uT)
    th = 2 * pi * np.random.random()
    q = np.vstack((np.array([[cos(th)]]), sin(th)*uT.T))
    if q[0,0] < 0:
        q *= -1
    print('q:')
    print(q)
    print(np.linalg.norm(q))
    R = q2R(q)
    print('R:')
    print(R)
    print(np.linalg.det(R))
    print('q:')
    q2 = R2q(R)
    print(q2)
    print(np.linalg.norm(q2))

def test_stdev():
    n = 10000
    x = np.zeros((1, n))
    y = np.empty((1, n))
    for i in range(n):
        y[0, i] = np.random.normal(0, 1.0)
    getZeroMeanGaussianErrorStdDev(x, y, plot=True)

def test_squash(seed=144000):
    np.random.seed(seed)
    v = np.random.normal(0, 2, (3,1))
    # n = np.array([[1./sqrt(3)],[1./sqrt(3)],[1./sqrt(3)]])
    n = np.random.normal(0, 2, (3,1))
    n /= np.linalg.norm(n)
    print('v = ')
    print(v.T)
    print('n = ')
    print(n.T)
    print('v squashed onto n\'s plane:')
    print((squashTo2DPlane(v, n)).T)

if __name__ == '__main__':
    # test_SO3()
    test_SE3()
    # test_qR_convs()
    # test_stdev()
    # test_squash()
    # test_squash(4)
    # test_squash(100)
    # test_squash(999)