#ifndef PROJECT_DPGTCONVERSIONS_H
#define PROJECT_DPGTCONVERSIONS_H

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>
#include <Eigen/Dense>

bool transform_pose(tf::TransformListener* tf_listener,
                    std::string parent_frame_id, 
                    std::string child_frame_id, 
                    const geometry_msgs::Pose& input_pose, 
                    geometry_msgs::Pose& output_pose){
    geometry_msgs::PoseStamped tmp_input_pose, tmp_output_pose;
    tmp_input_pose.header.stamp = tmp_output_pose.header.stamp = ros::Time(0);
    tmp_input_pose.header.frame_id = child_frame_id;
    tmp_output_pose.header.frame_id = parent_frame_id;
    tmp_input_pose.pose = input_pose;
    std::cout<<"child_frame_id = "<<child_frame_id<<std::endl;
    try{
        tf_listener->waitForTransform(parent_frame_id, child_frame_id,ros::Time(0), ros::Duration(0.5));
        tf_listener->transformPose(parent_frame_id, tmp_input_pose, tmp_output_pose);
    }   
    catch (tf::TransformException &ex){
            ROS_ERROR_STREAM("Transform cloud failure: " << ex.what());
            return false;
    }
    output_pose = tmp_output_pose.pose;
    ROS_INFO_STREAM("output_pose: "<< tmp_output_pose);
    return true;
}

void MatrixToTransform( const Eigen::Matrix4d& Tm,
                        tf::StampedTransform& transform,
                        std::string parent_frame_id = "parent_link",
                        std::string child_frame_id = "child_link")
{
  tf::Vector3 origin;
  origin.setValue(static_cast<double>(Tm(0,3)),static_cast<double>(Tm(1,3)),static_cast<double>(Tm(2,3)));
  tf::Matrix3x3 tf3d;
  tf3d.setValue(static_cast<double>(Tm(0,0)), static_cast<double>(Tm(0,1)), static_cast<double>(Tm(0,2)), 
                static_cast<double>(Tm(1,0)), static_cast<double>(Tm(1,1)), static_cast<double>(Tm(1,2)), 
                static_cast<double>(Tm(2,0)), static_cast<double>(Tm(2,1)), static_cast<double>(Tm(2,2)));
  transform = tf::StampedTransform(tf::Transform(tf3d, origin), ros::Time::now(),parent_frame_id,child_frame_id );
}

void MatrixToTransform( const Eigen::Matrix4d& Tm,
                        tf::Transform& transform)
{
  tf::Vector3 origin;
  origin.setValue(static_cast<double>(Tm(0,3)),static_cast<double>(Tm(1,3)),static_cast<double>(Tm(2,3)));
  tf::Matrix3x3 tf3d;
  tf3d.setValue(static_cast<double>(Tm(0,0)), static_cast<double>(Tm(0,1)), static_cast<double>(Tm(0,2)), 
                static_cast<double>(Tm(1,0)), static_cast<double>(Tm(1,1)), static_cast<double>(Tm(1,2)), 
                static_cast<double>(Tm(2,0)), static_cast<double>(Tm(2,1)), static_cast<double>(Tm(2,2)));
  transform = tf::Transform(tf3d, origin);
}

void print4x4Matrix(const Eigen::Matrix4d &matrix) {
  printf("Rotation matrix :\n");
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
  printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
  printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
  printf("Translation vector :\n");
  printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}

void TransformToMatrix(const tf::Transform& transform, 
                       Eigen::Matrix4d& transform_matrix) 
{
    Eigen::Translation3d tl_btol(
    transform.getOrigin().getX(), 
    transform.getOrigin().getY(), 
    transform.getOrigin().getZ());
    double roll, pitch, yaw;
    tf::Matrix3x3(transform.getRotation()).getEulerYPR(yaw, pitch, roll);
    Eigen::AngleAxisd rot_x_btol(roll, Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd rot_y_btol(pitch, Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd rot_z_btol(yaw, Eigen::Vector3d::UnitZ());
    transform_matrix = (tl_btol * rot_z_btol * rot_y_btol * rot_x_btol).matrix();
}

tf::Quaternion euler2Quaternion(const double roll, const double pitch, const double yaw)  
{  
    tf::Quaternion q;
    q.setRPY(roll, pitch, yaw);
    std::cout << "Euler2Quaternion result is:" <<std::endl;  
    std::cout << "x = " << q.x() <<std::endl;  
    std::cout << "y = " << q.y() <<std::endl;  
    std::cout << "z = " << q.z() <<std::endl;  
    std::cout << "w = " << q.w() <<std::endl<<std::endl;  
    return q;  
}  
  
Eigen::Vector3d Quaterniond2Euler(const double x,const double y,const double z,const double w)  
{  
    tf::Quaternion quat;
    geometry_msgs::Pose pose;
    pose.orientation.x = x;
    pose.orientation.y = y;
    pose.orientation.z = z;
    pose.orientation.w = w;

    tf::quaternionMsgToTF(pose.orientation, quat);

    double roll, pitch, yaw;//定义存储r\p\y的容器
    tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);//进行转换 

    Eigen::Vector3d euler;
    euler[2] = yaw;
    euler[1] = pitch;
    euler[0] = roll;

    // std::cout << "Quaterniond2Euler result is:" <<std::endl;  
    // std::cout << "roll = "<< euler[0] / M_PI * 180 << std::endl ;  
    // std::cout << "pitch = "<< euler[1] / M_PI * 180 << std::endl ;  
    // std::cout << "yaw = "<< euler[2] / M_PI * 180 << std::endl << std::endl;  
    return euler;
} 



#endif