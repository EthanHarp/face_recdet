Next steps:
Change from a service to an action for continuous usage
Create a launch file


Libraries used: Pytorch, OpenCV, ROS2, facenet-pytorch


For privacy reasons I will not include my dataset or model as that is directly based off of my identity

colcon build
source /opt/ros/humble/setup.bash
. install/setup.bash
ros2 run face_recdet service
ros2 run face_recdet client