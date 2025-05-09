#!/usr/bin/env python3
import rospy
from ackermann_msgs.msg import AckermannDrive
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import os
import yaml
import math
import tf
import numpy as np
import pandas as pd
from collections import deque

class F1TenthRacer:
    def __init__(self):
        # Load parameters
        abs_pkg = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(abs_pkg, '..', 'config', 'params.yaml')
        
        with open(config_path, 'r') as file:
            self.params = yaml.safe_load(file)
        
        # Load waypoints
        csv_path = os.path.join(abs_pkg, '..', 'csv', 'gp_centerline.csv')
        df = pd.read_csv(csv_path)
        self.waypoints = np.column_stack((df["x"].values, df["y"].values))
        
        # Race parameters
        self.L = 0.324  # Wheelbase
        self.MIN_LOOKAHEAD = 1.0  # meters
        self.MAX_LOOKAHEAD = 3.0  # meters
        self.LOOKAHEAD_GAIN = 0.25
        self.BASE_SPEED = 3.0
        self.MAX_SPEED = 8.0
        self.MIN_SPEED = 1.0
        self.MAX_STEERING_ANGLE = 0.4
        self.STEERING_FILTER_GAIN = 0.3
        self.MAX_STEERING_RATE = 0.2
        self.WAYPOINT_REACHED_DIST = 0.5  # meters
        self.CHECKPOINT_INTERVAL = 100
        
        # Race state
        self.current_goal_idx = 0
        self.race_started = False
        self.last_steering_angle = 0.0
        self.last_time = rospy.Time.now()
        self.car_position = np.array([0.0, 0.0])
        self.checkpoints_passed = set()
        self.steering_history = deque(maxlen=5)
        
        # Initialize ROS
        rospy.init_node("f1racing", anonymous=False)
        self.pub = rospy.Publisher(self.params["command_topic"], AckermannDrive, queue_size=10)
        rospy.Subscriber(self.params["odom_topic"], Odometry, self.odom_callback)
        rospy.Subscriber("/race_start", Bool, self.race_start_callback)
        
    def distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)
    
    def get_lookahead_point(self, current_pos, current_idx, speed):
        """Find lookahead point based on current speed"""
        lookahead_dist = self.MIN_LOOKAHEAD + self.LOOKAHEAD_GAIN * speed
        lookahead_dist = min(lookahead_dist, self.MAX_LOOKAHEAD)
        
        lookahead_idx = current_idx
        max_search_idx = min(current_idx + 50, len(self.waypoints) - 1)
        
        for i in range(current_idx, max_search_idx):
            if self.distance(current_pos, self.waypoints[i]) > lookahead_dist:
                lookahead_idx = i
                break
        
        if lookahead_idx == current_idx:
            lookahead_idx = min(current_idx + 1, len(self.waypoints) - 1)
        
        return lookahead_idx
    
    def calculate_steering(self, current_pos, yaw, goal_point):
        """Calculate pure pursuit steering angle"""
        dx = goal_point[0] - current_pos[0]
        dy = goal_point[1] - current_pos[1]
        rot_x = dx * math.cos(yaw) + dy * math.sin(yaw)
        rot_y = -dx * math.sin(yaw) + dy * math.cos(yaw)
        
        alpha = math.atan2(rot_y, rot_x)
        L_d = self.distance(current_pos, goal_point)
        return math.atan2(2.0 * self.L * math.sin(alpha), L_d)
    
    def smooth_steering(self, new_angle):
        """Apply smoothing to steering commands"""
        # Add to history
        self.steering_history.append(new_angle)
        
        # Low-pass filter using history
        smoothed = sum(self.steering_history) / len(self.steering_history)
        
        # Rate limiting
        max_change = self.MAX_STEERING_RATE * (rospy.Time.now() - self.last_time).to_sec()
        return np.clip(smoothed, 
                      self.last_steering_angle - max_change, 
                      self.last_steering_angle + max_change)
    
    def calculate_speed(self, steering_angle):
        """Calculate speed based on steering angle"""
        # Base speed with reduction for sharp turns
        speed = self.BASE_SPEED * (1.0 - 0.6 * abs(steering_angle)/self.MAX_STEERING_ANGLE)
        
        # Ensure within bounds
        return np.clip(speed, self.MIN_SPEED, self.MAX_SPEED)
    
    def race_start_callback(self, msg):
        """Handle race start signal"""
        if msg.data and not self.race_started:
            self.race_started = True
            rospy.loginfo("Race started!")
    
    def odom_callback(self, msg):
        """Main control loop"""
        if not self.race_started:
            return
            
        # Update car state
        self.car_position = np.array([msg.pose.pose.position.x, 
                                    msg.pose.pose.position.y])
        
        # Get orientation
        q = msg.pose.pose.orientation
        (_, _, yaw) = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Check if we've reached the end
        if self.current_goal_idx >= len(self.waypoints):
            rospy.loginfo("All waypoints completed!")
            self.pub.publish(AckermannDrive())  # Stop the car
            return
        
        # Get current goal point
        goal_point = self.waypoints[self.current_goal_idx]
        
        # Check if we've reached current waypoint
        if self.distance(self.car_position, goal_point) < self.WAYPOINT_REACHED_DIST:
            self.current_goal_idx += 1
            
            # Track checkpoints (every 100 waypoints)
            if (self.current_goal_idx % self.CHECKPOINT_INTERVAL) == 0:
                checkpoint_num = self.current_goal_idx // self.CHECKPOINT_INTERVAL
                self.checkpoints_passed.add(checkpoint_num)
                rospy.loginfo(f"Passed checkpoint {checkpoint_num}")
        
        # Find lookahead point
        lookahead_idx = self.get_lookahead_point(self.car_position, self.current_goal_idx, 
                                               self.BASE_SPEED)
        lookahead_point = self.waypoints[lookahead_idx]
        
        # Calculate steering
        new_steer_angle = self.calculate_steering(self.car_position, yaw, lookahead_point)
        smoothed_steer = self.smooth_steering(new_steer_angle)
        self.last_steering_angle = smoothed_steer
        
        # Calculate speed
        speed = self.calculate_speed(smoothed_steer)
        
        # Create and publish command
        cmd = AckermannDrive()
        cmd.steering_angle = np.clip(smoothed_steer, 
                                    -self.MAX_STEERING_ANGLE, 
                                    self.MAX_STEERING_ANGLE)
        cmd.speed = speed
        
        self.pub.publish(cmd)
        self.last_time = rospy.Time.now()

if __name__ == "__main__":
    racer = F1TenthRacer()
    rospy.spin()