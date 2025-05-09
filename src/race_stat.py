#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDrive
import numpy as np
from tf.transformations import euler_from_quaternion
import math
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import time
from std_msgs.msg import Bool
import os


abs_pkg = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(abs_pkg, '..', 'csv', 'gp_centerline.csv')




class StatsNode:
    def __init__(self):
        self.current_pose = None
        self.xs = []
        self.ys = []
        rospy.Subscriber('/car_1/odom', Odometry, self.odom_callback)
        self.waypoints = np.loadtxt(csv_path, delimiter=',',skiprows=1)
        self.race_start = rospy.Publisher('/race_start', Bool, queue_size=1)
        # self.checkpoints = self.waypoints[::100]
        offset = 80
        self.checkpoints_idx = np.arange(0, len(self.waypoints), 100) + np.random.randint(-offset,offset)
        self.checkpoints_idx = np.unique(self.checkpoints_idx)
        self.checkpoints = self.waypoints[self.checkpoints_idx % len(self.waypoints)]
        self.checkpoints = np.vstack((self.checkpoints, self.waypoints[-1]))

        plt.plot(self.waypoints[:,0], self.waypoints[:,1], label='Waypoints', color='black')
        plt.scatter(self.checkpoints[:,0], self.checkpoints[:,1], label='Checkpoints', color='magenta', s=15)
        plt.title('Checkpoints')
        plt.legend()
        plt.show()

        self.rate = rospy.Rate(10)

    
    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        orientation = msg.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        (roll, pitch, yaw) = euler_from_quaternion(orientation_list)
        self.current_pose = (x, y, yaw)

        self.xs.append(x)
        self.ys.append(y)

    

    def run(self):
        xs = []
        ys = []
        valid = False

        print('Waiting for 3 seconds')
        time.sleep(3)
        self.race_start.publish(Bool(True))
        print('Starting lap')

        lap_start = time.perf_counter()
        while not rospy.is_shutdown():
            if np.linalg.norm(self.current_pose[0:2] - self.waypoints[-1]) < 0.5 and time.perf_counter() - lap_start > 15 and valid:
                print("Lap completed!")
                break

            # Check if car reached 2nd checkpoint
            if len(self.xs) > 0 and len(self.ys) > 0:
                xs = np.array(self.xs)
                ys = np.array(self.ys)
                wps = np.array([xs, ys]).T
                distances = np.linalg.norm(wps - self.checkpoints[1], axis=1)
                min_distance = np.min(distances)
                if min_distance < 0.5:
                    valid = True
            

            if (time.perf_counter() - lap_start) > 100:
                print("Lap timed out!")
                break

            self.rate.sleep()

        
        end_time = time.perf_counter()
        lap_time = end_time - lap_start
        checkpoint_distances = []
        for checkpoint in self.checkpoints:
            checkpoint = np.around(checkpoint, 3)
            xs = self.xs
            ys = self.ys
            wps = np.array([xs, ys]).T
            distances = np.linalg.norm(wps - checkpoint, axis=1)
            min_distance = np.min(distances)
            checkpoint_distances.append([checkpoint[0], checkpoint[1], min_distance])

        total_error = 0
        for cp in checkpoint_distances:
            total_error += cp[2]
        total_error /= len(checkpoint_distances)

        plt.scatter(self.waypoints[:,0][::10], self.waypoints[:,1][::10], label='Waypoints', color='red', s=5)
        plt.scatter(self.checkpoints[:,0], self.checkpoints[:,1], label='Checkpoints', color='magenta', s=15)
        plt.plot(self.xs, self.ys, label='Path', color='blue')
        plt.title(f'Lap Time: {lap_time:.2f} seconds \n Average Error: {total_error:.2f} m')
        plt.legend()
        plt.show()


        

    

    
    

if __name__ == "__main__":
    rospy.init_node('racer_node', anonymous=True)
    racer_node = StatsNode()
    try:
        racer_node.run()
    except rospy.ROSInterruptException:
        pass