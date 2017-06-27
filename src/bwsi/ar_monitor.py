#!/usr/bin/env python
import rospy
from ar_track_alvar_msgs.msg import AlvarMarkers
import numpy as np
import tf
from tf.transformations  import compose_matrix

from convert_functions import *


class ArMonitor:
    def __init__(self, tf_listener=None, cb_func=None):
        self.markers = {}
        if tf_listener is None:
            self.tf_listener = tf.TransformListener()
        else:
            self.tf_listener = tf_listener
        self.T = compose_matrix(angles=(0,np.pi/2., 0))
        self.Tc_b =pos_and_quat_to_mat([-.39,-0.06,-0.09], [0,0,0,1] )
        self.br = tf.TransformBroadcaster()
        self.cb_func = cb_func
        self.sub = rospy.Subscriber("/ar_pose_marker", AlvarMarkers, self.cb, queue_size=1)
    def cb(self, msg):
        temp_markers = {}
        for marker in msg.markers:
            T_marker = pose_to_mat(marker.pose.pose)
            T_correct= self.T*T_marker*self.Tc_b
            pos, quat =  mat_to_pos_and_quat(T_correct)
            quat = (0, 0, 0, 1) # we know it is upright
            pos = (pos[0], pos[1], 0)
            pose = (pos,quat)
            temp_markers[marker.id] = pos
            self.broadcast( pose, 'correct_marker_%d' %marker.id, 'base_link')
        self.markers = temp_markers
        if self.cb_func is not None:
            self.cb_func(temp_markers)
    

    

    def broadcast(self, (pos,quat), link_from, link_to):
        time = rospy.Time.now()
        self.br.sendTransform( pos,quat,time, link_from, link_to)


if __name__=="__main__":
    rospy.init_node("ar_monitor_node")
    am = ArMonitor()
    rospy.spin()
