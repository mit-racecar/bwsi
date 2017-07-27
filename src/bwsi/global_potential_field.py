#!/usr/bin/env python

"""
Title: global_potential_field.py
Author: Ariel Anders and 
BeaverWorksSummerInsitute

This program contains functionality to load maps and display 
potential fields through RVIZ.
"""

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Pose, PoseStamped, Vector3, \
        Quaternion, Point, PointStamped
from std_msgs.msg import Header,ColorRGBA
import rospy
import numpy as np
import cv_bridge
from itertools import product
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from ackermann_msgs.msg import AckermannDriveStamped as Drive

to_quat =lambda a:Quaternion(*quaternion_from_euler(0,0,a))
to_angle=lambda q:euler_from_quaternion((q.x,q.y,q.z,q.w))[2]
to_pose=lambda x,y,th:Pose(Point(x,y,0),to_quat(th))

class ARMap:
    def __init__(self, ground, obstacles, ar_tags):
        self.GROUND = ground
        self.OBSTACLES = obstacles
        self.AR_TAGS = ar_tags
        self.map_pub = rospy.Publisher('simple_map', MarkerArray, queue_size=1)
        self.vf_pub = rospy.Publisher('vector_field', MarkerArray, queue_size=1)
        self.COLORS = {'ground': (1,1,1,1), 'obs':(.3,.3,.3,1), 'ar':(0,0,0,1)}
        self.THICKNESS = {'ground':.0001, 'obs':1, 'ar':.2}
        rospy.sleep(.5)
        self.drawMap()
        rospy.loginfo("Done initializing global potential field.  View RVIZ ")
    
    """
    ================  Drawing functions below =====================
    """
    
    def clearMarkers(self, maps=False, vector=False):
        ma = MarkerArray()
        ma.markers= [self.stamp(Marker())]
        ma.markers[0].action = 3
        ma.markers[0].ns="maps"
        if maps: self.map_pub.publish(ma)
        if vector: self.vf_pub.publish(ma)

    def drawMap(self):
        self.clearMarkers(maps=True)
        ma = MarkerArray()
        ma.markers = [self.drawRect('ground',0,0,0, *self.GROUND)]
        for i,o in enumerate(self.OBSTACLES):
            ma.markers.append(self.drawRect('obs',i,*o))
        for i, (x,y,th) in enumerate(self.AR_TAGS):
            ma.markers.append(self.drawRect('ar', i, x,y,.1,.1))
        self.map_pub.publish(ma)
    
    def drawVectorField(self, potential_func, goal=None):
        self.clearMarkers(vector=True)
        ma = MarkerArray()
        x = np.linspace(0.05, self.GROUND[0], 30)
        y = np.linspace(0.05, self.GROUND[1], 30)

        for pt in product(x,y):
            r,th = potential_func(pt)
            ma.markers.append(self.drawArrow(pt[0],pt[1],r,th))
        if goal is not None:
            ma.markers.append(self.drawGoal(goal))
        self.vf_pub.publish(ma)

    def stamp(self, msg):
        msg.header = Header(0, rospy.Time.now(), 'map')
        return msg

    def drawArrow(self, x, y, r, th):
        r = max(.2, min(1, r *0.01))
        m = self.stamp(Marker())
        m.id = hash("%s%s"%(x,y))%10000
        m.type = 0
        m.action = 0
        m.ns = "maps"
        x2,y2 = x+ r*np.cos(th), y+r*np.sin(th)
        m.points =[ Point(x,y,0), Point(x2,y2,0)]
        m.scale=Vector3(.025, .05, 0.1)
        m.color = ColorRGBA(1,0,0,1)
        return m

    def drawGoal(self, goal):
        m = self.stamp(Marker())
        m.pose = to_pose(goal[0], goal[1], 0)
        m.type = 2
        m.color = ColorRGBA(0,1,0,1)
        m.scale = Vector3(.5,.5,.5)
        m.ns ="maps"
        m.id=hash('goal')%1000
        return m

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

    def clearMarkers(self, maps=False, vector=False):
        ma = MarkerArray()
        ma.markers= [self.stamp(Marker())]
        ma.markers[0].action = 3
        ma.markers[0].ns="maps"
        if maps: self.map_pub.publish(ma)
        if vector: self.vf_pub.publish(ma)

   



class GlobalPotentialField:
    K_a = 1  # gain for attractive potentials
    K_r = 3 # gain for repulsive potentials
    Obs_ignore = 2  #if car is more than .75m away from vehicle ignore it
    Obs_gamma = 2   # factor for how much we care about obstacles 

    def __init__(self, armap):
        self.goal = None
        self.armap=armap
        self.LINES = [self.lineSegments(*obs) for obs in armap.OBSTACLES]
        self.LINES += [self.lineSegments(0,0,*armap.GROUND)]
        self.goal_sub = rospy.Subscriber('clicked_point', \
                PointStamped,self.goalCB,queue_size=1)
        self.armap.drawVectorField(self.potential, self.goal)
        rospy.loginfo("Done initializing global potential field.  View RVIZ ")
    
    def goalCB(self, msg):
        self.goal = np.array([msg.point.x, msg.point.y])
        rospy.loginfo( "updated goal to %s" % self.goal )
        self.armap.drawVectorField(self.potential, self.goal)
    
    
    def potential_attract(self, pt):
        """
        #XXX For students to do
        this function should return a numpy array of 2 values
        u = (delta x, and delta y)

        u should be directly correlated with how far the point pt
        is from self.goal
        """
        if self.goal is None:
            return np.array([0,0])
        else:
            e = self.goal - np.array(pt)
            u = e*(0.5*np.linalg.norm(e)**2) 
            return u

    def potential_repulse(self, pt):
        """
        #XXX For students to do
        this function should return a numpy array of 2 values
        u = (delta x, and delta y)
        
        u should be computed by looking at the distance from every
        obstacle.  Obstacles that are close to the point pt should have
        a higher repulsive force than obstacles far away.
        """
        
        # For each obstacle, dists will contain: (r, (dx,dy))
        # r is the minimum perpendicular distance from the obstacle 
        # r = ||dx,dy || 
        # (dx,dy) is the vector that shows the direction the pt is 
        # respulsed from the obstacle
        # dists has a separeate item for the walls and each obstacle.

        dists = [self.distanceToPoly(pt, l) for l in self.LINES]

        def get_u(r, vector):
            if r<=1e-5 or r > self.Obs_ignore:
                return np.array([0,0])
            scale = (1.0/self.Obs_gamma)*((1.0/r)**self.Obs_gamma)
            return scale*vector

        u_s = np.array([get_u(d, v) for (d,v)  in dists])
        return np.sum(u_s, axis=0)

    def potential(self, pt):
        """
        This function takes a weighted average of the attractive and repulsive
        forces for point pt.

        then, it computes the polar coordinate which is then used for graphing
        the potential field

        (Note: This function is already completed)
        """
        a = self.potential_attract(pt)
        r = self.potential_repulse(pt)
        u = self.K_a*a + self.K_r*r
        th = np.arctan2(u[1], u[0])
        r = np.linalg.norm(u)
        return r,th


    """
    ==========Functions for computing distance to obstacles  =========
    """

    def distancePtToSegment(self,pt, s):
        """
        This function computes the minimum distance from a point to a
        line segment.
        If the point is past the line, then it computes the distance
        to an end point.  Otherwise, it computes the perpendicular
        distance to the line
        """
        pt = np.array(pt)
        s = np.array(s[0]), np.array(s[1])
        dist= lambda x,y: (np.linalg.norm(x-y), x-y)
        v = s[1] - s[0]
        w = pt - s[0]
        c1 = np.dot(w, v)
        if c1 <= 0: return dist(pt, s[0])
        c2 = np.dot(v,v)
        if c2 <= c1: return dist(pt, s[1])
        b = c1/c2
        pb = s[0] + b*v
        return dist(pt, pb)

    def distanceToPoly(self, pt, lines):
        dists = [self.distancePtToSegment(pt,l) for l in lines]
        dists = sorted(dists, key=lambda x: x[0])
        return dists[0]

    def lineSegments(self, x,y,w,h):
        points = np.array([ (0,0), (w,0), (w,h), (0, h)])
        points = points + np.array([x,y])
        lines = [(points[i-1], points[i])for i in range(len(points))]
        return np.array(lines, dtype=np.float32)


class CarControl:
    K_speed = 1
    K_angle = 1
    def __init__(self, gpf):
        self.gpf = gpf
        self.iter = 0
        self.cmd = None
        self.pub_cmd = rospy.Publisher("/vesc/ackermann_cmd_mux/input/teleop", Drive, queue_size=1)

        self.pub_vis = rospy.Publisher("potential_field_cmd", \
                PoseStamped, queue_size=1)
        #XXX using simulatorself.sub_pose = rospy.Subscriber('/expected_pose', \
        self.sub_pose = rospy.Subscriber('/mle_pose', \
                PoseStamped, self.cbPose,queue_size=1)
        self.run()
    
    def run(self):
        while not rospy.is_shutdown():
            if self.cmd is not None:
                self.pub_cmd.publish(self.cmd)
            rospy.sleep(0.01)

    def cbPose(self, msg):
            
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = to_angle(msg.pose.orientation)
        
        potential = self.gpf.potential((x,y))
        
        angle = np.arctan2(potential[1],potential[0]) - z

        debug = ""
        if np.abs(angle) > np.pi/2.:
            angle = -angle
            speed = -1
            debug += "go backwards"
        else:
            speed = 1

        msg.pose.orientation = to_quat(angle)
        self.pub_vis.publish(msg)
        dr = Drive()
        dr.header = msg.header

                    
        dr.drive.steering_angle = self.K_angle*(angle)
        dr.drive.speed = self.K_speed*speed
        if False : #potential[0]==0 and potential[1] == 0:
            dr.drive.steering_angle = 0
            dr.drive.speed = 1
            self.cmd = dr

            debug += "No potential value"
        elif x < 0 or y < 0 or x > self.gpf.armap.GROUND[0] or y > self.gpf.armap.GROUND[1]:
            self.cmd = None
            debug += "Car is out of bounds"
        else:
            self.cmd = dr



        self.iter = (self.iter +1) % 1000
        if self.iter%10 == 0:
            #print  (x,y,z), 
            if dr.drive.steering_angle == 0:
                pass
            elif dr.drive.steering_angle > 0:
                debug += " left"
            else:
                debug += " right"
            print "(%.2f, %.2f), %.2f %s" %( potential[0], potential[1], angle*180./np.pi, debug)


          

if __name__=="__main__":
    rospy.init_node("global_potential_field")
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
 
    
    armap = ARMap(GROUND, OBSTACLES, AR_TAGS)
    gpf = GlobalPotentialField(armap)
    cc = CarControl(gpf)

    rospy.spin()



