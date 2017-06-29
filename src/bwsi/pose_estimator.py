import rospy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Point, Quaternion, TwistWithCovariance,\
    Twist, Vector3, PoseStamped, PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Header
import tf
import tf.transformations
from ar_monitor import ArMonitor
from scipy.stats import norm


def sum_to_one(v):
    return v / sum(v)


def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def transform_matrix((x, y, theta)):
    c, s = np.cos(theta), np.sin(theta)
    T = [[c, -s, x],
         [s, c,  y],
         [0, 0,  1]
         ]
    return np.matrix(T)


def transform_to_pose(T):
    x, y, _ = T[:, 2]
    theta = np.arctan2(T[0, 1], T[0, 0])
    return x, y, theta


def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))


def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians."""
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return yaw


def get_pose(odom_msg):
    x = odom_msg.pose.pose.position.x
    y = odom_msg.pose.pose.position.y
    th = quaternion_to_angle(odom_msg.pose.pose.orientation)
    return np.array([x, y, th])


def particle_to_pose(particle):
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose


def make_header(frame_id, stamp=None):
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header


def particles_to_poses(particles):
    return map(particle_to_pose, particles)


def pose_to_pos_quat(pose):
    pos = (pose[0], pose[1], 0)
    o = angle_to_quaternion(pose[2])
    quat = o.x, o.y, o.z, o.w
    return pos, quat


class PoseEstimator:
    ar_locations = {1: (1, 0, 0)}

    def __init__(self, particles=500, init_pose=(0., 0., 0.)):
        self.shape = (particles, 3)
        self.num_particles = particles
        self.particle_range = np.arange(particles)
        self.particles = np.array([init_pose] * particles)
        self.weights = (1. / particles) * np.ones((particles))
        self.alphas = [1e-6] * 4  # TODO let these be modified
        rospy.loginfo("pose estimator is initialized")
        self.prev_odom = None
        self.pose_pub = rospy.Publisher("mle_pose", PoseStamped, queue_size=1)
        self.belief_pub = rospy.Publisher("belief", PoseArray, queue_size=1)
        self.odom_cb = rospy.Subscriber("/vesc/odom/", Odometry,
                                        self.cb_odom, queue_size=1)
        self.pose_sub = rospy.Subscriber("/initialpose",
                                         PoseWithCovarianceStamped, self.clicked_pose, queue_size=1)

        self.br = tf.TransformBroadcaster()
        self.ar_monitor = ArMonitor(cb_func=self.cb_obs)

    def cb_obs(self, markers):
        for i, delta in markers.items():
            if not i in self.ar_locations:
                continue
            ar_pose = self.ar_locations[i]
            curr_pose = [(m - d) for (m, d) in zip(ar_pose, delta)]
            self.mcl(None, curr_pose)

    def cb_odom(self, data):
        if self.prev_odom == None:
            self.prev_odom = data
            return
        u = [get_pose(self.prev_odom), get_pose(data)]
        self.prev_odom = data
        self.mcl(u, None)

    def normal(self, deviation):
        return np.random.normal(loc=0.0,
                                scale=deviation,
                                size=self.num_particles)

    def clicked_pose(self, msg):
        '''
        Receive pose messages from RViz and initialize 
        the particle distribution in response.
        '''
        rospy.loginfo("SETTING POSE")
        pose = msg.pose.pose
        self.weights = np.ones(self.num_particles) / float(self.num_particles)

        self.particles[:, 0] = pose.position.x + self.normal(.5)
        self.particles[:, 1] = pose.position.y + self.normal(.5)
        self.particles[:, 2] = quaternion_to_angle(
            pose.orientation) + self.normal(.4)

    def publish_pose(self):
        self.mle_pose = np.dot(self.particles.transpose(), self.weights)
        pa = PoseArray()
        pa.header = make_header("odom")
        pa.poses = particles_to_poses(self.particles)
        self.belief_pub.publish(pa)
        pos, quat = pose_to_pos_quat(self.mle_pose)
        msg = PoseStamped()
        msg.header = pa.header
        msg.pose = Pose(Point(*pos), Quaternion(*quat))

        self.pose_pub.publish(msg)
        #self.br.sendTransform(pos,quat,rospy.Time.now(),"base_link", "pf_odom")

    def mcl(self, u, z):
        if u is not None:
            self.sample_motion_model(u)
        if z is not None:
            self.measurement_model(z)
            self.inject_samples(z)
            self.reweight_samples()
        self.publish_pose()
        return

    def sample_motion_model(self, u):
        # add relative movement from odom to particles

        # the following computes lines 2-4 from algorithm
        # x_(t-1) = px, py, pth and x_t =  cx, cy,cth
        (px, py, pth), (cx, cy, cth) = u

        delta_rot1 = np.arctan2(cy - py, cx - px) - pth
        delta_trans = np.sqrt((px - cx)**2 + (py - cy)**2)
        delta_rot2 = cth - pth - delta_rot1

        # compute samples for values around the deltas
        # lines 5-7
        a1, a2, a3, a4 = self.alphas
        r1 = delta_rot1**2.
        r2 = delta_rot2**2.
        t = delta_trans**2.
        dev1 = a1 * r1 + a2 * t
        dev2 = a3 * t + a4 * r1 + a4 * r2
        dev3 = a1 * r2 + a2 * t
        if dev1 == 0: dev1 = 0.1
        if dev2 == 0: dev2 = 0.1
        if dev3 == 0: dev3 = 0.1

        if dev1 == 0:
            rot1_samples = np.ones(self.num_particles)*delta_rot1
        else:
            rot1_samples = delta_rot1 - \
            np.random.normal(0., dev1, self.num_particles)
        if dev2 == 0:
            trans_samples = np.ones(self.num_particles)*delta_trans
        else:
            trans_samples = delta_trans - \
            np.random.normal(0., dev2, self.num_particles)
        if dev3 == 0:
            rot2_samples = np.ones(self.num_particles)*delta_rot2
        else:
            rot2_samples = delta_rot2 - \
            np.random.normal(0., dev3, self.num_particles)

        # lines 8-10

        th = self.particles[:, 2]
        deltas = np.zeros((self.num_particles, 3))
        deltas[:, 0] = trans_samples * np.cos(th + rot1_samples)
        deltas[:, 1] = trans_samples * np.sin(th + rot1_samples)
        deltas[:, 2] = rot1_samples + rot2_samples

        self.particles += deltas
        return

    def measurement_model(self, z):
        if z is None:
            return
        x = self.particles[:, 0]
        y = self.particles[:, 1]
        px = norm.pdf(x, scale=2.0, loc=z[0])
        py = norm.pdf(y, scale=2.0, loc=z[1])
        self.weights = sum_to_one(np.multiply(px, py))
        return

    def inject_samples(self, z):
        unlikely = self.weights < 0.1 / self.num_particles
        random = np.random.uniform(0, 1, self.num_particles) < .3
        inserts = np.where(np.logical_or(unlikely, random))
        self.particles[inserts] = z
        self.weights[inserts] = .01 / self.num_particles
        self.weights = sum_to_one(self.weights)

    def reweight_samples(self):
        samples = np.random.choice(
            a=self.particle_range,
            size=self.num_particles,
            replace=True,
            p=self.weights
        )
        self.particles = self.particles[samples]
        self.weights = sum_to_one(self.weights[samples])


if __name__ == "__main__":
    rospy.init_node("pose_estimator")
    pe = PoseEstimator()
    rospy.spin()
