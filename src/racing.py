#!/usr/bin/env python3
import rospy
from ackermann_msgs.msg import ackermann_msgs,AckermannDrive
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
import os
import yaml
import math
import tf   
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



abs_pkg = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(abs_pkg, '..', 'config', 'params.yaml')

with open(config_path, 'r') as file:
    parameter = yaml.safe_load(file)
    odom_topic = parameter["odom_topic"]
    print(odom_topic)
    command_topic = parameter["command_topic"]
    print(command_topic)

csv_path = os.path.join(abs_pkg, '..', 'csv', 'gp_centerline.csv')


df = pd.read_csv(csv_path)
print(df.head())  # Show first 5 rows
# print(df["x"][1])


target_points_x = df["x"]
target_points_y = df["y"]

rospy.init_node("f1racing", anonymous=False)
pub = rospy.Publisher(command_topic, AckermannDrive, queue_size=10)
K_p = 0.6
L = 0.324
current_goal_idx = 0
race_starting = False
path_x = []
path_y = []
time2 = 0



def angle_alpha(y_rel, x_rel):
    ans  = 0
    
    ans = math.atan2(y_rel, x_rel)
    return ans



def distance_ld(curr_pos,my_target):
    goal_x = my_target[0]
    goal_y = my_target[1]
    pos_x = curr_pos[0]
    pos_y = curr_pos[1]
    ans = math.sqrt((goal_x - pos_x) ** 2 + (goal_y - pos_y) ** 2)
    return ans


def plot_trajectory():
    plt.figure()
    plt.plot(path_x, path_y, label="Car Trajectory", linewidth=2)
    plt.title("Car Pure Pursuit Trajectory")
    plt.title(f"Lap time = {(time2 - time1).to_sec():.2f} seconds")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.legend()
    plt.savefig("car_trajectory.png")
    plt.show()  



rate = rospy.Rate(10)



def lets_start(data):
    print(data.data)
    global race_starting
    check = data.data
    race_starting = check


def callback(data):
    global time2
    global current_goal_idx, race_starting
    if current_goal_idx >= len(target_points_x):
        rospy.loginfo("All goals reached!")
        if time2 == 0:
            time2 = rospy.Time.now()
        pub.publish(AckermannDrive()) 
        plot_trajectory()
        return
        
    curr_x = data.pose.pose.position.x
    curr_y = data.pose.pose.position.y
    cmd = AckermannDrive()
    curr_pos = [curr_x, curr_y]
    
    # Get next 20 points (or remaining points if near end)
    lookahead = min(20, len(target_points_x) - current_goal_idx)
    check_x = np.array(target_points_x[current_goal_idx:current_goal_idx+lookahead])
    check_y = np.array(target_points_y[current_goal_idx:current_goal_idx+lookahead])
    
    # Calculate average curvature over the lookahead points
    def calculate_curvatures(x_points, y_points):
        curvatures = []
        for i in range(1, len(x_points)-1):
            # Using three-point curvature estimation
            x0, x1, x2 = x_points[i-1], x_points[i], x_points[i+1]
            y0, y1, y2 = y_points[i-1], y_points[i], y_points[i+1]
            
            dx1, dy1 = x1 - x0, y1 - y0
            dx2, dy2 = x2 - x1, y2 - y1
            
            # Cross product to estimate curvature
            cross = abs(dx1 * dy2 - dx2 * dy1)
            denom = (dx1**2 + dy1**2)**1.5
            if denom > 1e-6:  # Avoid division by zero
                curvatures.append(cross / denom)
        return curvatures
    
    curvatures = calculate_curvatures(check_x, check_y)
    
    if len(curvatures) > 0:
        # Weighted average (more weight to nearer points)
        weights = np.linspace(1.0, 0.5, len(curvatures))
        avg_curvature = np.average(curvatures, weights=weights)
        
        # Dynamic speed adjustment based on curvature
        max_speed = 4.0  # Maximum speed on straight sections
        min_speed = 1.0   # Minimum speed on sharp turns
        
        # Smooth speed transition based on curvature
        speed = max_speed / (1.0 + 2.0 * avg_curvature)
        speed = max(min(speed, max_speed), min_speed)
        
        rospy.loginfo(f"Avg curvature: {avg_curvature:.4f}, Speed: {speed:.2f}")
    else:
        speed = 3.0  # Default speed if curvature can't be calculated
    
    # Rest of your pure pursuit logic remains the same
    path_x.append(curr_x)
    path_y.append(curr_y)
    goal = (target_points_x[current_goal_idx], target_points_y[current_goal_idx])
    
    if distance_ld(curr_pos, goal) < 0.5:
        current_goal_idx += 1
    
    # Calculate steering angle (your existing code)
    q = data.pose.pose.orientation
    orientation_list = [q.x, q.y, q.z, q.w]
    (_, _, yaw) = tf.transformations.euler_from_quaternion(orientation_list)
    dx = goal[0] - curr_x
    dy = goal[1] - curr_y
    alpha = (math.atan2(dy, dx) - yaw)
    L_d = distance_ld(goal, curr_pos)
    steer_angle = math.atan2(2 * L * math.sin(alpha), L_d)
    
    cmd.steering_angle = -1 * max(min(steer_angle, 0.5), -0.5)
    cmd.speed = speed
    
    if race_starting:
        pub.publish(cmd)

def listen():
    rospy.Subscriber(odom_topic, Odometry, callback)
    rospy.Subscriber("/race_start",Bool,lets_start)
    rospy.spin()


if __name__ == "__main__":
    # time1 = rospy.rostime("now")
    time1 = rospy.Time.now()
    listen()