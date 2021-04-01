## @package pyvitools
#  Select tools written in Python for processing and analyzing data for visual-inertial estimation applications.
#
#  Provides tools for:
#
#   * Algebraic anipulation of vectors and SO(3)/SE(3) objects.
#   * Differentiating and integrating the above.
#   * ROSbag processing.
#   * Dataset manipulation.
#   * (Coming soon) Plotting of all the above.


# Base packages
import numpy as np
import cv2
from math import sqrt, atan2, cos, sin, pi

# Import point cloud visualization if available
try:
    import open3d
    _PCL_vis = True
except ImportError:
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
    _ROS_msgs = False

# Import ROSbag parser if available
try:
    import rospy, rosbag
    _ROS_bag = True
except ImportError:
    _ROS_bag = False

# Import ROS cv bridge if available
try:
    from cv_bridge import CvBridge
    _ROS_bridge = True
except ImportError:
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

    def __init__(self, arr=None, qw=1.0, qx=0.0, qy=0.0, qz=0.0):
        if arr is None:
            self.arr = np.array([[qw, qx, qy, qz]]).T
        else:
            if not isinstance(arr, np.ndarray) and arr.shape != (4,1):
                raise TypeError('SO(3) default constructor requires a 4x1 numpy array quaternion [qw qx qy qz]^T.')
            self.arr = arr / np.linalg.norm(arr)
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
            return np.array([[0, -o[2], o[1]],
                             [o[2], 0, -o[0]],
                             [-o[1], o[0], 0]])
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
    def fromTranslationAndRotation(*args):
        if len(args) == 1:
            return SE3(args[0])
        elif len(args) == 2 and isinstance(args[1], np.ndarray):
            return SE3(np.vstack((args[0], args[1])))
        elif len(args) == 2 and isinstance(args[1], SO3):
            return SE3(np.vstack((args[0], args[1].q())))
        else:
            raise TypeError('Translation/Quaternion allowed types are (7x1 array), (3x1 array, 4x1 array), (3x1 array, SO3).')

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
            Jl_inv = np.eye(3) - 0.5 * W + (1 - th * cos(th/2) / (2 * sin(th/2))) / (th*th) * (W*W)
        else:
            Jl_inv = np.eye(3)

        return np.vstack((np.dot(Jl_inv, X.t), w))

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
            Jl = np.eye(3) + (1-cos(th))/(th*th)*W + (th-sin(th))/(th*th*th) * (W*W)
        else:
            Jl = np.eye(3)

        return SE3.fromTranslationAndRotation(np.dot(Jl, rho), q)

##################################################################
########################### PID TOOLS ############################

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
                # print(t)
                # print(msg)
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
    #  @param truth_topic (Optional) Topic to take truth values from. If not specified, will look for topics with message type Odometry, Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped, Transform, TransformStamped, Point, or PointStamped.
    #  @param image_topic (Optional) Topic to take images from. If not specified, will look for topics with message type Image.
    #  @param imu_topic (Optional) Topic to take IMU measurements from. If not specified, will look for topics with message type Imu.
    #  @param cam_info_topic (Optional) Topic to take camera info from. If not specified, will look for topics with message type CameraInfo. 
    def __init__(self, bagfile, truth_topic=None, image_topic=None, imu_topic=None, cam_info_topic=None):
        self.truth  = ROSbagStateDataset(bagfile, truth_topic)
        self.camera = ROSbagCameraDataset(bagfile, image_topic, cam_info_topic)
        self.imu    = ROSbagImuDataset(bagfile, imu_topic)

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

    ## Applies a rigid body transform to all truth state fields, resulting in a transformed dataset.
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
            v_BW_B = np.array(self.truth.vel_data[:,i:i+1], copy=True)
            w_BW_B = np.array(self.truth.omg_data[:,i:i+1], copy=True)

            self.truth.pos_data[:,i:i+1] = np.dot(q_N_W.inverse().R(), p_BW_W - t_NW_W + np.dot(q_B_W.R(), t_UB_B))
            self.truth.att_data[:,i:i+1] = (q_N_W.inverse() * q_B_W * q_U_B).arr
            self.truth.vel_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), v_BW_B + np.cross(w_BW_B, t_UB_B, axis=0))
            self.truth.omg_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), w_BW_B)

    ## Transforms the velocity truth fields from the world frame to the body frame using the attitude truth fields.
    #
    #  Assumes that the attitude truth fields represent \f$R_B^W\f$ and are of the form \f$\begin{bmatrix}q_w & q_x & q_y & q_z\end{bmatrix}^{\top}\f$.
    def rotateVelocityTruthToBody(self):
        print('Rotating truth velocities into the body frame.')
        for i in range(self.truth.t_data.shape[1]):
            q_B_W = SO3(np.array(self.truth.att_data[:,i:i+1], copy=True))
            v_BW_W = np.array(self.truth.vel_data[:,i:i+1], copy=True)

            self.truth.vel_data[:,i:i+1] = np.dot(q_B_W.inverse().R(), v_BW_W)

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

    ## Numerically differentiates the truth pose fields to obtain body-frame translational and anglar velocities.
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

    ## Write the current dataset contents to a new ROSbag file.
    #  @param bagfile The path to the new bagfile.
    #  @param truth_topic New topic name for the truth nav_msgs/Odometry field.
    #  @param image_topic New topic name for the sensor_msgs/Image field.
    #  @param imu_topic New topic name for the sensor_msgs/Imu field.
    #  @param cam_info_topic New topic name for the sensor_msgs/CameraInfo field.
    def writeBag(self, bagfile, truth_topic='/truth', image_topic='/rgb', imu_topic='/imu', cam_info_topic='/rgb/info'):
        self.truth.write(bagfile, topic_name=truth_topic)
        self.camera.write(bagfile, img_topic_name=image_topic, info_topic_name=cam_info_topic, spec='a')
        self.imu.write(bagfile, topic_name=imu_topic, spec='a')

def test_SO3():
    R1 = SO3.random()
    print(R1)
    R2 = R1 + np.array([[0.5, 0.2, 0.1]]).T
    print(R2)
    print(R2-R1)

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

if __name__ == '__main__':
    # test_SO3()
    test_qR_convs()