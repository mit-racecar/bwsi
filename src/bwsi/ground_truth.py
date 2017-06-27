#!/usr/bin/env python

# publish ground truth odometry when using simulator
# author ariel anders


import rospy
from gazebo_msgs.msg import ModelStates
import tf
from convert_functions import pose_to_pos_and_quat

class Odom:
    def __init__(self):
        self.name = 'racecar'
        self.link_from = 'base_link'
        self.link_to ="odom"
        self.br = tf.TransformBroadcaster()
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb, queue_size=1)
    
    def cb(self, data):
        time = rospy.Time.now()
        link_to = self.link_to

        for i in range(len(data.name)):
            if self.name == data.name[i]:
                link_from = self.link_from
            else:
                link_from = data.name[i]
            self.broadcast(data.pose[i], time, link_from, link_to)


    def broadcast(self, pose, time, link_from, link_to):
        pos,quat = pose_to_pos_and_quat(pose)
        self.br.sendTransform( pos,quat,time, link_from, link_to)


if __name__=="__main__":
    rospy.init_node("ground_truth_odom")
    Odom()
    rospy.spin()
