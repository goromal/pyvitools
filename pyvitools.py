# Base packages
import numpy as np
import cv2
from math import sqrt, atan2, cos, sin

# Import ROS message types if available
try:
    from geometry_msgs.msg import Quaternion, QuaternionStamped
    from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovariance, PoseWithCovarianceStamped
    from geometry_msgs.msg import Transform, TransformStamped
    from geometry_msgs.msg import Point, PointStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CameraInfo, Image, Imu
    __ROS_msgs = True
except ImportError:
    __ROS_msgs = False

# Import ROSbag parser if available
try:
    import rospy, rosbag
    __ROS_bag = True
except ImportError:
    __ROS_bag = False

# Import ROS cv bridge if available
try:
    from cv_bridge import CvBridge
    __ROS_bridge = True
except ImportError:
    __ROS_bridge = False

class ROSNotFoundError(Exception):
    pass

##################################################################
######################## MANIFOLD OBJECTS ########################

def q2R(q):
    qw, qx, qy, qz = q[0,0], q[1,0], q[2,0], q[3,0]
    qxx = qx*qx
    qxy = qx*qy
    qxz = qx*qz
    qyy = qx*qy
    qyz = qy*qz
    qzz = qz*qz
    qwx = qw*qx
    qwy = qw*qy
    qwz = qw*qz
    return np.array([[1-2*qyy-2*qzz, 2*qxy-2*qwz, 2*qxz+2*qwy],
                     [2*qxy+2*qwz, 1-2*qxx-2*qzz, 2*qyz-2*qwx],
                     [2*qxz-2*qwy, 2*qyz+2*qwx, 1-2*qxx-2*qyy]])

def R2q(R):
    t = np.trace(R)
    q = np.zeros((4,1))
    if t > 1.0:
        q[0] = t
        q[1] = R[2,1]-R[1,2]
        q[2] = R[0,2]-R[2,0]
        q[3] = R[1,0]-R[0,1]
    else:
        i, j, k = 0, 1, 2
        if R[1,1] > R[0,0]:
            i, j, k = 1, 2, 0
        if R[2,2] > R[i,i]:
            i, j, k = 2, 0, 1
        t = R[i,i]-(R[j,j]+R[k,k])+1.0
        q[0] = R[k,j]-R[j,k]
        q[i+1] = t
        q[j+1] = R[i,j]+R[j,i]
        q[k+1] = R[k,i]+R[i,k]
    q *= 0.5 / sqrt(t)

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
        if not __ROS_msgs:
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

    def R(self):
        return q2R(self.arr)

    def q(self):
        return self.arr

    def inverse(self):
        return SO3(qw=self.arr[0,0], qx=-self.arr[1,0], qy=-self.arr[2,0], qz=-self.arr[3,0])

    def invert(self):
        self.arr[1:,:] *= -1.0

    def toQuaternionMsg(self):
        if not __ROS_msgs:
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
        if not __ROS_msgs:
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
        if not __ROS_msgs:
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
            return SE3(np.vstack(args[0], args[1].q()))
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
        if not __ROS_msgs:
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
        if not __ROS_msgs:
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
########################## ROSBAG PARSE ##########################

class ROSbagParserBase(object):
    def __init__(self, bagfile):
        if not __ROS_bag:
            raise ROSNotFoundError('ROSbag parser not found on system.')
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

    def process_msgs_by_type(self, conv_f, msg_type, topic_name=None):
        data = None
        for topic, msg, t in self.bag.read_messages():
            if isinstance(msg, msg_type) and ((not topic_name is None and topic == topic_name) or topic_name is None):
                datum = np.vstack((np.array([[t.to_sec()-self.t0]]), conv_f(msg)))
                data = (datum if data is None else np.hstack((data, datum)))
        return data

    def close(self):
        self.bag.close()

class ROSbagCameraDataset(ROSbagParserBase):
    def __init__(self, bagfile, img_topic_name=None, info_topic_name=None):
        if not __ROS_bridge:
            raise ROSNotFoundError('Cannot create ROSbag camera dataset without cv_bridge.')
        super(ROSbagCameraDataset, self).__init__(bagfile)
        self.img_topic = img_topic_name
        self.info_topic = info_topic_name
        
        self.t_data = None
        self.img_data = list()
        self.h = None
        self.w = None
        self.dist_model = None
        self.D = None
        self.K = None
        self.R = None
        self.P = None

        self.bridge = CvBridge()

        for topic, msg, t in self.bag.read_messages():
            if isinstance(msg, Image) and ((not self.img_topic is None and topic == self.img_topic) or self.img_topic is None):
                t_datum = np.array([[t.to_sec()-self.t0]])
                self.t_data = (t_datum if self.t_data is None else np.hstack((self.t_data, t_datum)))
                self.img_data.append(self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough'))
            elif isinstance(msg, CameraInfo) and ((not self.info_topic is None and topic == self.info_topic) or self.info_topic is None) and self.h is None:
                self.h = msg.height
                self.w = msg.width
                self.dist_model = msg.distortion_model
                self.D = np.array(msg.D)
                self.K = np.array(msg.K).reshape((3,3))
                self.R = np.array(msg.R).reshape((3,3))
                self.P = np.array(msg.P).reshape((3,4))

        self.close()

    def write(self, bagfile, img_topic_name='/camera/raw', info_topic_name='/camera/info'):
        bag = rosbag.Bag(bagfile, 'w')
        try:
            for i in range(self.t_data.size):
                t = rospy.Time(self.t_data[0,i])
                cam_msg = self.bridge.cv2_to_imgmsg(self.img_data[i], encoding='passthrough')
                bag.write(img_topic_name, cam_msg, t)
                info_msg = CameraInfo()
                info_msg.height = self.h
                info_msg.width = self.w
                info_msg.distortion_model = self.dist_model
                info_msg.D = list(self.D)
                info_msg.K = list(self.K.reshape((1,9))[0])
                info_msg.R = list(self.R.reshape((1,9))[0])
                info_msg.P = list(self.P.reshape((1,12))[0])
                bag.write(info_topic_name, info_msg, t)
        finally:
            bag.close()

class ROSbagImuDataset(ROSbagParserBase):
    def __init__(self, bagfile, topic_name=None):
        super(ROSbagImuDataset, self).__init__(bagfile)
        self.topic = topic_name

        self.t_data = None
        self.acc_data = None
        self.gyr_data = None

        data = self.process_msgs_by_type(self.convImu, Imu, topic_name)
        if data != None:
            self.t_data = data[0:1,:]
            self.acc_data = data[1:4,:]
            self.gyr_data = data[4:7,:]

        self.close()

    def write(self, bagfile, topic_name='/imu'):
        bag = rosbag.Bag(bagfile, 'w')
        try:
            for i in range(self.t_data.size):
                t = rospy.Time(self.t_data[0,i])
                msg = Imu()
                msg.linear_acceleration.x = self.acc_data[0,i]
                msg.linear_acceleration.y = self.acc_data[1,i]
                msg.linear_acceleration.z = self.acc_data[2,i]
                msg.angular_velocity.x = self.gyr_data[0,i]
                msg.angular_velocity.y = self.gyr_data[1,i]
                msg.angular_velocity.z = self.gyr_data[2,i]
                bag.write(topic_name, msg, t)
        finally:
            bag.close()

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

        data = self.process_msgs_by_type(self.convOdometry, Odometry, topic_name)
        if data != None:
            self.t_data = data[0:1,:]
            self.pos_data = data[1:4,:]
            self.att_data = data[4:8,:]
            self.vel_data = data[8:11,:]
            self.omg_data = data[11:14,:]
            self.close()
            return

        data = self.process_msgs_by_type(self.convPose, Pose, topic_name)
        data = (self.process_msgs_by_type(self.convPoseStamped, PoseStamped, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convPoseWithCovariance, PoseWithCovariance, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convPoseWithCovarianceStamped, PoseWithCovarianceStamped, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convTransform, Transform, topic_name) if data is None else data)
        data = (self.process_msgs_by_type(self.convTransformStamped, TransformStamped, topic_name) if data is None else data)
        if data != None:
            self.t_data = data[0:1,:]
            self.pos_data = data[1:4,:]
            self.att_data = data[4:8,:]
            self.close()
            return

        data = self.process_msgs_by_type(self.convPoint, Point, topic_name)
        data = (self.process_msgs_by_type(self.convPointStamped, PointStamped, topic_name) if data is None else data)
        if data != None:
            self.t_data = data[0:1,:]
            self.pos_data = data[1:4,:]
            
        self.close()

    def write(self, bagfile, topic_name='/state'):
        bag = rosbag.Bag(bagfile, 'w')
        try:
            for i in range(self.t_data.size):
                t = rospy.Time(self.t_data[0,i])
                msg = Odometry()
                if self.pos_data != None:
                    msg.pose.pose.position.x = self.pos_data[0,i]
                    msg.pose.pose.position.y = self.pos_data[1,i]
                    msg.pose.pose.position.z = self.pos_data[2,i]
                if self.att_data != None:
                    msg.pose.pose.orientation.w = self.att_data[0,i]
                    msg.pose.pose.orientation.x = self.att_data[1,i]
                    msg.pose.pose.orientation.y = self.att_data[2,i]
                    msg.pose.pose.orientation.z = self.att_data[3,i]
                if self.vel_data != None:
                    msg.twist.twist.linear.x = self.vel_data[0,i]
                    msg.twist.twist.linear.y = self.vel_data[1,i]
                    msg.twist.twist.linear.z = self.vel_data[2,i]
                if self.omg_data != None:
                    msg.twist.twist.angular.x = self.omg_data[0,i]
                    msg.twist.twist.angular.y = self.omg_data[1,i]
                    msg.twist.twist.angular.z = self.omg_data[2,i]
                bag.write(topic_name, msg, t)
        finally:
            bag.close()

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

class ROSbagSingleAgentViMonoSlamDataset(object):
    def __init__(self, bagfile, truth_topic=None, image_topic=None, imu_topic=None, cam_info_topic=None):
        '''
        ROSbagSingleAgentViMonoSlamDataset(bagfile, [truth_topic], [image_topic], [imu_topic], [cam_info_topic])

        Instantiates a dataset from bagfile, containing:
        - Truth state data
        - Camera image and info data
        - IMU data

        The optional topic name specifications will tell the dataset which topics to poll for each data field.
        Else, it will search for a topic that has the appropriate message type.
        '''
        self.truth  = ROSbagStateDataset(bagfile, truth_topic)
        self.camera = ROSbagCameraDataset(bagfile, image_topic, cam_info_topic)
        self.imu    = ROSbagImuDataset(bagfile, imu_topic)

    def transformTruthFrame(self, T):
        q_U_B = T.q
        t_UB_B = T.t
        for i in range(len(self.truth.t_data)):
            p_BW_W = np.array(self.truth.pos_data[:,i:i+1], copy=True)
            q_B_W = SO3(np.array(self.truth.att_data[:,i:i+1], copy=True))
            v_BW_B = np.array(self.truth.vel_data[:,i:i+1], copy=True)
            w_BW_B = np.array(self.truth.omg_data[:,i:i+1], copy=True)

            self.truth.pos_data[:,i:i+1] = p_BW_W + np.dot(q_B_W.R(), t_UB_B)
            self.truth.att_data[:,i:i+1] = (q_B_W * q_U_B).arr
            self.truth.vel_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), v_BW_B + np.cross(w_BW_B, t_UB_B))
            self.truth.omg_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), w_BW_B)

    def transformImuFrame(self, T):
        # https://notes.andrewtorgesen.com/doku.php?id=public:imu
        q_U_B = T.q
        t_UB_B = T.t
        wd = Differentiator((3,1))
        t_prev = None
        for i in range(len(self.imu.t_data)):
            t = self.imu.t_data[0,i]
            Ts = (t-t_prev if i > 0 else 0)
            t_prev = t
            a = np.array(self.imu.acc_data[:,i:i+1], copy=True)
            w = np.array(self.imu.gyr_data[:,i:i+1], copy=True)

            O = SO3.hat(wd.calculate(w, Ts)) + np.dot(w, w.T) - np.dot(w.T, w) * np.eye(3)
            self.imu.acc_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), a + np.dot(O, t_UB_B))
            self.imu.gyr_data[:,i:i+1] = np.dot(q_U_B.inverse().R(), w)

    def synthesizeImu(self, g=np.array([[0,0,-9.81]]).T, sigma_eta=(0,0), kappa=(0,0), sigma_beta=(0,0)):
        # g is 3x1 vector in W frame, default arg assumes that z_W points up
        # https://notes.andrewtorgesen.com/doku.php?id=public:imu

        if self.truth.t_data != None:
            imu_t = list(self.truth.t_data)
        else:
            raise Exception('Truth data required for IMU synthesis.')
        n = len(imu_t)

        if self.truth.att_data != None:
            imu_q = [SO3(np.array(q, copy=True)) for q in np.hsplit(self.truth.att_data, n)]
            if self.truth.omg_data != None:
                print('Synthesizing rate gyro data from body-frame angular velocity truth.')
                imu_w = [np.array(w, copy=True) for w in np.hsplit(self.truth.omg_data, n)]
            else:
                print('Synthesizing rate gyro data from attitude truth.')
                imu_w = list()
                w_B = Differentiator((3,1))
                t_prev = None
                for i in range(n):
                    t  =imu_t[i]
                    Ts = (t-t_prev if i > 0 else 0)
                    t_prev = t
                    imu_w.append(w_B.calculate(SO3(self.truth.att_data[:,i:i+1]), Ts))
        else:
            raise Exception('Some kind of attitude truth data required for IMU synthesis.')

        if self.truth.vel_data != None:
            print('Syntheszing accel data from body-frame velocity truth.')
            imu_v_W = list()
            for i in range(n):
                imu_v_W.append(np.dot(imu_q[i].R(), np.array(self.truth.vel_data[:,i:i+1], copy=True)))
        elif self.truth.pos_data != None:
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

    def writeBag(self, bagfile, truth_topic='/truth', image_topic='/rgb', imu_topic='/imu', cam_info_topic='/rgb/info'):
        self.truth.write(bagfile, topic_name=truth_topic)
        self.camera.write(bagfile, img_topic_name=image_topic, info_topic_name=cam_info_topic)
        self.imu.write(bagfile, topic_name=imu_topic)

if __name__ == '__main__':
    R1 = SO3.random()
    R2 = R1 + np.array([[0.5, 0.2, 0.1]]).T
    print(R2-R1)