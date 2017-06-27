import numpy as np
import tf.transformations


# convert a Pose message to a 4x4 np matrix
def pose_to_mat(pose):
    quat = [pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w]
    pos = np.matrix([pose.position.x, pose.position.y, pose.position.z]).T
    mat = np.matrix(tf.transformations.quaternion_matrix(quat))
    mat[0:3, 3] = pos
    return mat


# convert a 4x4 np matrix to a Pose message
def mat_to_pose(mat):
    pose = Pose()
    pose.position.x = mat[0, 3]
    pose.position.y = mat[1, 3]
    pose.position.z = mat[2, 3]
    quat = tf.transformations.quaternion_from_matrix(mat)
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    return pose


# convert a tf transform to a 4x4 np mat
def transform_to_mat(transform):
    quat = [transform.rotation.x, transform.rotation.y,
            transform.rotation.z, transform.rotation.w]
    pos = np.matrix([transform.translation.x, transform.translation.y,
                     transform.translation.z]).T
    mat = np.matrix(tf.transformations.quaternion_matrix(quat))
    mat[0:3, 3] = pos
    return mat


# convert a 4x4 np matrix to position and quaternion lists
def mat_to_pos_and_quat(mat):
    quat = tf.transformations.quaternion_from_matrix(mat).tolist()
    pos = mat[0:3, 3].T.tolist()[0]
    return (pos, quat)

# convert a (pos,quat)  to a 4x4 np matrix


def pos_and_quat_to_mat(pos, quat):
    mat = np.matrix(tf.transformations.quaternion_matrix(quat))
    mat[0, 3] = pos[0]
    mat[1, 3] = pos[1]
    mat[2, 3] = pos[2]
    return mat

# convert a pose message to pos and quat tuples


def pose_to_pos_and_quat(pose):
    quat = [pose.orientation.x, pose.orientation.y,
            pose.orientation.z, pose.orientation.w]
    pos = [pose.position.x, pose.position.y, pose.position.z]
    return pos, quat

# convert a pose message to a list


def pose_to_list(pose):
    pos, quat = pose_to_pos_and_quat(pose)
    return list(pose) + list(quat)

# get the 4x4 transformation matrix from frame1 to frame2 from TF


def get_transform(tf_listener, frame1, frame2):
    temp_header = Header()
    temp_header.frame_id = frame1
    temp_header.stamp = rospy.Time(0)
    try:
        frame1_to_frame2 = tf_listener.asMatrix(frame2, temp_header)
    except:
        rospy.logerr("tf transform was not there between %s and %s" %
                     (frame1, frame2))
        return np.matrix(np.identity(4))
    return np.matrix(frame1_to_frame2)
