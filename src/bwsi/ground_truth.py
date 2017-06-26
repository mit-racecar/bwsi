#!/usr/bin/env python

# publish ground truth odometry when using simulator
# author ariel anders


import rospy
from gazebo_msgs.msg import ModelStates
import tf

class Odom:
    def __init__(self):
        self.name = 'racecar'
        self.link_from = 'base_link'
        self.link_to ="odom"
        self.br = tf.TransformBroadcaster()
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self.cb, queue_size=1)
    def cb(self, data):
        if not self.name  in data.name:
            return
        i = data.name.index(self.name)
        
        p = data.pose[i].position
        q = data.pose[i].orientation
        pos = (p.x, p.y, p.z)
        quat = (q.x, q.y, q.z, q.w)

        self.br.sendTransform( pos,quat,rospy.Time.now(),self.link_from,self.link_to)
    
if __name__=="__main__":
    rospy.init_node("ground_truth_odom")
    Odom()
    rospy.spin()
