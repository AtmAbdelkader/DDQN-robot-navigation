# DDQN-robot-navigation

To prepare your PC you need:
* Install Ubuntu 20.04 on PC or in Virtual Machine
Download the ISO [Ubuntu 20.04](https://ubuntu.com/download/alternative-downloads) for your PC
* Install [ROS Noetic](http://wiki.ros.org/noetic/Installation/Ubuntu) on your Ubuntu 20.04
* Install ROS missing libraries. Some libraries that are used in this project are not in the standard ROS package. Install them with:
```sh
sudo apt-get update && sudo apt-get install -y \
     ros-noetic-ros-controllers \
     ros-noetic-gazebo-ros-control \
     ros-noetic-joint-state-publisher-gui \
     ros-noetic-joy \
     ros-noetic-joy-teleop \
     ros-noetic-turtlesim \
     ros-noetic-robot-localization \
     ros-noetic-actionlib-tools
```
