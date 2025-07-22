cmake_minimum_required(VERSION 2.8.3)
project(navigation)

ADD_COMPILE_OPTIONS(-std=c++14 )

find_package(yaml-cpp REQUIRED)

find_package(catkin REQUIRED COMPONENTS
    actionlib
    geometry_msgs
    message_generation
    move_base_msgs
    nav_msgs
    roscpp
    rospy
    sensor_msgs
    std_msgs
    tf2_eigen
    tf2_geometry_msgs
    tf2_ros
    roslib
    std_srvs
    tf
)



include_directories(
	${catkin_INCLUDE_DIRS} 
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  include)

# add_message_files(
#   FILES
#   Pose6D.msg
# )

generate_messages(
 DEPENDENCIES
 geometry_msgs
 std_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime nav_msgs tf
  DEPENDS EIGEN3 PCL
  INCLUDE_DIRS
)

# catkin_install_python(PROGRAMS
#   src/lcm_server.py
#   src/exlcm/example_t.py
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

add_executable(nav_to_object src/nav_to_object.cpp)
target_link_libraries(nav_to_object
  ${catkin_LIBRARIES}
)

add_dependencies(nav_to_object ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} )
target_link_libraries(nav_to_object
  ${catkin_LIBRARIES} ${YAML_CPP_LIBRARIES}
)

