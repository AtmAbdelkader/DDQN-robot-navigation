# DDQN-robot-navigation

Deep Reinforcement Learning for mobile robot navigation in ROS Gazebo simulator. Using Double Deep Q_learning (DDQN) and Prioritized experience replay buffer with DDQN, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained in ROS Gazebo simulator with PyTorch.


## Installation

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

Main dependencies: 
* [PyTorch](https://pytorch.org/get-started/locally/)
* [Tensorboard](https://github.com/tensorflow/tensorboard)


Clone the repository:
```shell
cd ~
### Clone this repo
git clone https://github.com/AtmAbdelkader/DDQN-robot-navigation.git
```

Compile the workspace:
```shell
cd ~/DDQN-robot-navigation/catkin_ws
### Compile
catkin_make
```

Open a terminal and set up sources:
```shell
export ROS_HOSTNAME=localhost
export ROS_MASTER_URI=http://localhost:11311
export ROS_PORT_SIM=11311
export GAZEBO_RESOURCE_PATH=~/DDQN-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
source ~/.bashrc
cd ~/DDQN-robot-navigation/catkin_ws
source devel/setup.bash
```

Run the training:
```shell
cd ~/DDQN-robot-navigation/DDQN
python3 train_ddqn.py
```

To check the training process on tensorboard:
```shell
cd ~/DRL-robot-navigation/DDQN
tensorboard --logdir runs
```

To kill the training process:
```shell
killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient python python3
```

Once training is completed, test the model:
```shell
cd ~/DRL-robot-navigation/DDQN
python3 test_ddqn.py
```
