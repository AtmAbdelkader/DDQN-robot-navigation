import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, DeleteModel
from geometry_msgs.msg import Twist, Pose
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
#from sensor_msgs.msg import LaserScan
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import threading

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.3


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    if -3.8 > x > -6.2 and 6.2 > y > 3.8:
        goal_ok = False

    if -1.3 > x > -2.7 and 4.7 > y > -0.2:
        goal_ok = False

    if -0.3 > x > -4.2 and 2.7 > y > 1.3:
        goal_ok = False

    if -0.8 > x > -4.2 and -2.3 > y > -4.2:
        goal_ok = False

    if -1.3 > x > -3.7 and -0.8 > y > -2.7:
        goal_ok = False

    if 4.2 > x > 0.8 and -1.8 > y > -3.2:
        goal_ok = False

    if 4 > x > 2.5 and 0.7 > y > -3.2:
        goal_ok = False

    if 6.2 > x > 3.8 and -3.3 > y > -4.2:
        goal_ok = False

    if 4.2 > x > 1.3 and 3.7 > y > 1.5:
        goal_ok = False

    if -3.0 > x > -7.2 and 0.5 > y > -1.5:
        goal_ok = False

    if x > 4.5 or x < -4.5 or y > 4.5 or y < -4.5:
        goal_ok = False

    return goal_ok

'''def check_pos(x, y):
    goal_ok = True

    if 1.5 < x < 2.5 and 0.5 < y < 1.5:
        goal_ok = False
    
    if -2.5 < x < -1.5 and 0.5 < y < 1.5:
        goal_ok = False

    if 1.5 < x < 2.5 and -0.5 > y > -1.5:
        goal_ok = False

    if -2.5 < x < -1.5 and -0.5 > y > -1.5:
        goal_ok = False
    
    return goal_ok'''


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.lidar = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

        #Dynamic obstacles :
        '''
         # متغيرات الحركة
        self.update_interval = 0.1  # الوقت بين التحديثات بالثواني
        self.box_speed = 0.5  # سرعة الحركة الحالية (متر/ثانية)
        self.box_acceleration = 0.1  # تسارع الصناديق (متر/ثانية^2)
        self.box_positions = {i: {'x': np.random.uniform(-5, 5), 'y': np.random.uniform(-5, 5),
                                  'vx': np.random.uniform(-self.box_speed, self.box_speed), 
                                  'vy': np.random.uniform(-self.box_speed, self.box_speed)} for i in range(4)}
        
        # حدود البيئة
        self.x_min, self.x_max = -5, 5
        self.y_min, self.y_max = -5, 5

        # بدء خيوط التحديث
        self.update_thread = threading.Thread(target=self.update_box_positions)
        self.update_thread.daemon = True
        self.update_thread.start()

    def update_box_positions(self):
        rate = rospy.Rate(1 / self.update_interval)  # تحديث حسب الفترة الزمنية المحددة
        while not rospy.is_shutdown():
            for i in range(4):
                name = "cardboard_box_" + str(i)
                box_state = ModelState()
                box_state.model_name = name

                # الحصول على بيانات الصندوق
                pos = self.box_positions[i]
                x, y = pos['x'], pos['y']
                vx, vy = pos['vx'], pos['vy']

                # تحديث السرعة
                vx += self.box_acceleration * self.update_interval
                vy += self.box_acceleration * self.update_interval

                # تحديث الموقع
                x += vx * self.update_interval
                y += vy * self.update_interval

                # تقييد حركة الصناديق ضمن الحدود
                if x < self.x_min or x > self.x_max:
                    vx = -vx
                    x = np.clip(x, self.x_min, self.x_max)
                if y < self.y_min or y > self.y_max:
                    vy = -vy
                    y = np.clip(y, self.y_min, self.y_max)

                # تحديث بيانات الصندوق
                self.box_positions[i] = {'x': x, 'y': y, 'vx': vx, 'vy': vy}

                # تعيين حالة النموذج في Gazebo
                box_state.pose.position.x = x
                box_state.pose.position.y = y
                box_state.pose.position.z = 0.0
                box_state.pose.orientation.x = 0.0
                box_state.pose.orientation.y = 0.0
                box_state.pose.orientation.z = 0.0
                box_state.pose.orientation.w = 1.0

                self.set_state.publish(box_state)
            
            rate.sleep()

        '''

    '''#Read Hukuyo LIDAR data :
    def scan_callback(self, msg):
        ranges = msg.ranges
        self.lidar_data = np.ones(self.environment_dim) * 10

        angle_increment = msg.angle_increment
        angle_min = msg.angle_min

        for i in range(len(ranges)):
            distance = ranges[i]
            if distance < msg.range_max:
                angle = angle_min + i * angle_increment
                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= angle < self.gaps[j][1]:
                        self.lidar_data[j] = min(self.lidar_data[j], distance)
                        break'''
            
    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False
        
        # Define the mapping from discrete actions to velocities
        action_mappings = {
            0: (1.0, 0.0),    # Move forward with a linear velocity of 1.0 and no angular velocity
            1: (0.0, 1.0),    # Turn left with no linear velocity and an angular velocity of 1.0
            2: (0.0, -1.0),   # Turn right with no linear velocity and an angular velocity of -1.0
            3: (0.85, 0.5),   # Move forward and turn left
            4: (0.85, -0.5)   # Move forward and turn right 
        }
        
        # Map the discrete action to linear and angular velocities
        if action in action_mappings:
            linear_vel, angular_vel = action_mappings[action]
        else:
            raise ValueError("Invalid action: {}".format(action))
        
        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel
        vel_cmd.angular.z = angular_vel
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action) 

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        #self.random_box()
        # Update box positions based on velocities

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            rospy.loginfo("\033[32mGOAL!!!\033[0m")
            target = True
            done = True

        robot_state = [distance, theta, linear_vel, angular_vel]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        return state, reward, done, target

    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        #x = -2.5
        #y = 0.0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        self.change_goal()
        # randomly scatter boxes in the environment
        self.random_box()
        #self.publish_markers([0.0, 0.0])

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):

        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False
        while not goal_ok:
            self.goal_x = random.uniform(self.upper, self.lower)
            self.goal_y = random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)
        
         # Delete the existing goal marker before spawning a new one
        self.delete_goal_marker()

        time.sleep(0.2)

        # Once a valid goal is found, visualize it in Gazebo
        self.spawn_goal_marker(self.goal_x, self.goal_y)

    def random_box(self):
            # Randomly change the location of the boxes in the environment on each reset to randomize the training
            # environment
            for i in range(4):
                name = "cardboard_box_" + str(i)

                x = 0
                y = 0
                box_ok = False
                while not box_ok:
                    x = np.random.uniform(-6, 6)
                    y = np.random.uniform(-6, 6)
                    box_ok = check_pos(x, y)
                    distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                    distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                    if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                        box_ok = False
                box_state = ModelState()
                box_state.model_name = name
                box_state.pose.position.x = x
                box_state.pose.position.y = y
                box_state.pose.position.z = 0.0
                box_state.pose.orientation.x = 0.0
                box_state.pose.orientation.y = 0.0
                box_state.pose.orientation.z = 0.0
                box_state.pose.orientation.w = 1.0
                self.set_state.publish(box_state)
    
    #Visualize goal point in Gazebo simulator
    def spawn_goal_marker(self, x, y):
        # Path to your SDF model file
        model_path = '/home/belabed/DRL-robot-navigation/TD3/gazebo_goal/goal.xml'
        with open(model_path, 'r') as f:
            model_xml = f.read()

        # Define the pose for the goal marker
        goal_pose = Pose()
        goal_pose.position.x = x
        goal_pose.position.y = y
        #goal_pose.position.z = z

        # Call the service to spawn the model
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            spawn_model_prox = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
            spawn_model_prox("goal_marker", model_xml, "", goal_pose, "world")
            #rospy.loginfo("Goal marker spawned successfully at ({}, {})!".format(x, y))
        except rospy.ServiceException as e:
            rospy.logerr("Failed to spawn goal marker: %s", e)
    
    def delete_goal_marker(self, model_name="goal_marker"):
        rospy.wait_for_service('/gazebo/delete_model')
        try:
            delete_model_prox = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
            delete_model_prox(model_name)
            #rospy.loginfo("Deleted existing goal marker: {}".format(model_name))
        except rospy.ServiceException as e:
            rospy.logwarn("Failed to delete goal marker: %s", e)

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)
        
    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            rospy.loginfo("\033[31mcollision is detected!!\033[0m")
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            #rospy.loginfo("reward 100")
            return 100.0
        elif collision:
            #rospy.loginfo("reward -100")
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action / 2 - abs(action) / 2 - r3(min_laser) / 2
