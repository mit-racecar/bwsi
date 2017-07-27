#!/usr/bin/env python

"""
Title: simple_map.py
Author: Ariel Anders
"""

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, Vector3, Quaternion, Point
from std_msgs.msg import Header,ColorRGBA
import rospy
import numpy as np
import cv_bridge

from tf.transformations import quaternion_from_euler, euler_from_quaternion

to_quat =lambda a:Quaternion(*quaternion_from_euler(0,0,a))
to_angle=lambda q:euler_from_quaternion((q.x,q.y,q.z,q.w))[2]
to_pose=lambda x,y,th:Pose(Point(x,y,0),to_quat(th))


def dist_pt_to_segment(pt, s):
    pt = np.array(pt)
    s = np.array(s[0]), np.array(s[1])
    dist= lambda x: np.linalg.norm(x)

    v = s[1] - s[0]
    w = pt - s[0]
    
    c1 = np.dot(w, v)
    if c1 <= 0: return dist(pt, s[0])
    c2 = np.dot(v,v)
    if c2 <= c1: return dist(pt, s[1])
    b = c1/c2
    pb = s[0] + b*v
    return dist(pt, pb)
"""
def line( (x1,y1), (x2,y2)):
        a = y1-y2
        b = x2-x1
        c = (x1-x2)*y1 + (y2-y1)*x1
        v = np.array([a,b,c])
        v = v/(np.sqrt(a**2 + b**2))
        return v

def get_square_eqs(x,y,w,h):
    points = np.array([ (0,0), (w,0), (w,h), (0, h)])
    points = points + np.array([x,y])
    lines = [line(points[i-1], points[i])for i in range(len(points))]
    return np.array(lines, dtype=np.float32)

def perp_distance(pt, eqs):
    dists = np.dot(pt,eqs[:,:2].T) + eqs[:,2]
    if all(dists >0) or all(dists <0):
        return 0
    else:
        return abs(dists)
"""

class SimpleMap:
    COLORS = {'ground': (1,1,1,1), 'obs':(.3,.3,.3,1), 'ar':(0,0,0,1)}
    THICKNESS = {'ground':.0001, 'obs':1, 'ar':.2}

    def __init__(self, ground, obstacles, ar_tags):

        self.GROUND = ground
        self.OBSTACLES = obstacles
        self.AR_TAGS = ar_tags
        self.map_pub = rospy.Publisher('simplemap', MarkerArray, queue_size=1)
        self.vf_pub = rospy.Publisher('vector_field', MarkerArray, queue_size=1)
        rospy.sleep(.5)
        self.drawMap()
    
    def stamp(self, msg):
        msg.header = Header(0, rospy.Time.now(), 'odom')
        return msg

    def drawRect(self, name, index, x, y, w, h):
        m = self.stamp(Marker())
        m.id =hash(name+str(index))%100
        m.type = 1
        m.action = 0
        m.ns = "maps"
        m.pose = to_pose(x+w/2.,y+h/2.,0)
        z = self.THICKNESS[name]
        m.scale = Vector3(w,h,z)
        m.color = ColorRGBA(*self.COLORS[name])
        return m

    def drawMap(self):
        ma = MarkerArray()
        ma.markers = [self.drawRect('ground',0,0,0, *self.GROUND)]
        for i,o in enumerate(self.OBSTACLES):
            ma.markers.append(self.drawRect('obs',i,*o))
        for i, (x,y,th) in enumerate(self.AR_TAGS):
            ma.markers.append(self.drawRect('ar', i, x,y,.1,.1))
        self.map_pub.publish(ma)

        


if __name__=="__main__":
    rospy.init_node("simple_map")
    GROUND = (4.80,   6.95)

    OBSTACLES = [
        #( x,    y,      w,      h)
        (1.25,  1.80,   1.90,   1.50),
        (1.90,  4.73,   1.32,   2.22),
        (0.,    5.41,   1.13,   1.54),
        ]
    AR_TAGS = [
        (0., 0., 0),
        (1.0, 1.0, 0),
        (2.0,2.0, 0),
        ]
 
    
    #sm = SimpleMap(GROUND, OBSTACLES, AR_TAGS)
    #rospy.spin()



