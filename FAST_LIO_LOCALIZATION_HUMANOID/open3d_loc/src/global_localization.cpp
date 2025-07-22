#include <iostream>
#include <queue>
#include <thread>
#include <mutex>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <geometry_msgs/TransformStamped.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <std_msgs/Float32.h>

#include <eigen_conversions/eigen_msg.h>
// #include <pcl/common/transforms.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <open3d/Open3D.h>

#include "open3d_registration/open3d_registration.h"
#include "open3d_conversions/open3d_conversions.h"

#define PI 3.1415926

class KalmanFilter
{
public:
    KalmanFilter()
    {
    }

    void KalmanFilterInit(double processVar, double estimatedMeasVar, double posteriEstimate = 0.0, double posteriErrorEstimate = 1.0)
    {
        processVar_ = processVar;
        estimatedMeasVar_ = estimatedMeasVar;
        posteriEstimate_ = posteriEstimate;
        posteriErrorEstimate_ = posteriErrorEstimate;
    }
    void inputLatestNoisyMeasurement(double measurement)
    {
        double prioriEstimate = posteriEstimate_;
        double prioriErrorEstimate = posteriErrorEstimate_ + processVar_;

        double blendingFactor = prioriErrorEstimate / (prioriErrorEstimate + estimatedMeasVar_);
        posteriEstimate_ = prioriEstimate + blendingFactor * (measurement - prioriEstimate);
        posteriErrorEstimate_ = (1 - blendingFactor) * prioriErrorEstimate;
    }

    double getLatestEstimatedMeasurement()
    {
        return posteriEstimate_;
    }

private:
    double processVar_;
    double estimatedMeasVar_;
    double posteriEstimate_;
    double posteriErrorEstimate_;
};

class GloabalLocalization
{
private:
    /* data */
public:
    GloabalLocalization(ros::NodeHandle &nh, ros::NodeHandle &nh_private);
    ~GloabalLocalization();

    /// @brief 初始化定位
    void LocalizationInitialize();

    /// @brief 订阅fast_lio里程计信息
    void CallbackBaselink2Odom(const nav_msgs::Odometry::ConstPtr &baselink2odom);
    /// @brief 订阅在baselink下的点云
    void CallbackScan(const sensor_msgs::PointCloud2::ConstPtr &scan_in_baselink);

    /// @brief 订阅在初始位姿
    void CallbackInitialPose(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &initialpose);

    void StartLoc();

    void Localization();

    /// @brief 欧拉角转mat3x3
    /// @param euler
    /// @return
    Eigen::Matrix3d Euler2Matrix3d(const Eigen::Vector3d euler);

    /// @brief 获取tf关系到矩阵
    /// @param frame_id
    /// @param child_frame_id
    /// @param matrix
    /// @return
    bool GetTfTransformToMatrix(
        std::string frame_id, std::string child_frame_id, Eigen::Matrix4d &matrix);

    /// @brief compute 3d distance between two points
    /// @param a
    /// @param b
    /// @return 距离值
    double ComputeMotionDis(const Eigen::Vector3d &a, const Eigen::Vector3d &b);

private:
    ros::NodeHandle nh_;
    ros::NodeHandle nh_private_;

    /// @brief 订阅baselink2odom,即fast_lio的里程计信息
    ros::Subscriber sub_baselink2odom_;

    /// @brief 订阅当前帧点云
    ros::Subscriber sub_scan_cur_;

    /// @brief 订阅初始位姿
    ros::Subscriber sub_initialpose_;

    /// @brief baselink到odom的pose表达
    nav_msgs::Odometry pose_baselink2odom_;

    /// @brief bselink到odom的变换矩阵表达
    Eigen::Matrix4d mat_baselink2odom_;
    /// @brief odom到map的矩阵
    Eigen::Matrix4d mat_odom2map_;
    Eigen::Matrix4d mat_odom2map_kalman_;
    /// @brief baselink到map = mat_odom2map * mat_baselink2odom
    Eigen::Matrix4d mat_baselink2map_;
    /// @brief initialpose初始位姿
    Eigen::Matrix4d mat_initialpose_;

    std::mutex lock_mat_odom2map_;

    /// @brief baselink和运动中心
    Eigen::Matrix4d mat_baselink2motionlink_;

    /// @brief imulink到baselink
    Eigen::Matrix4d mat_imulink2baselink_;

    /// @brief 初始位姿, x, y, z, roll, pitch, yaw (单位:度degrees)
    std::vector<float> initialpose_;

    /// @brief 原始地图点云
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_ori_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_coarse_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_fine_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_map_cur_;
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan_cur_;

    std::queue<open3d::geometry::PointCloud> que_pcd_scan_;
    int queue_maxsize_;
    double voxelsize_coarse_;
    double voxelsize_fine_;

    /// @brief 定位配准fitness(overlap)阈值
    double threshold_fitness_;
    /// @brief 配准fitness(overlap)阈值
    double threshold_fitness_init_;

    std::thread thread_loc_;
    std::mutex lock_scan_;
    std::mutex lock_exit_;
    bool flag_exit_;

    ros::Publisher pub_baselink2map_;
    ros::Publisher pub_baselink2map_kalman_;
    ros::Publisher pub_motionlink2map_;
    ros::Publisher pub_odom2map_;
    ros::Publisher pub_odom2map_kalman_;
    ros::Time timestamp_odom_;
    std::mutex lock_timestamp_;

    ros::Publisher pub_map_;
    ros::Publisher pub_scan_;
    ros::Publisher pub_scan2map_;
    ros::Publisher pub_submap_;
    ros::Publisher pub_localization_3d_;
    ros::Publisher pub_localization_3d_confidence_;
    ros::Publisher pub_localization_3d_delay_ms_;

    geometry_msgs::PoseStamped localization_3d_;
    std_msgs::Float32 localization_3d_confidence_;
    std_msgs::Float32 localization_3d_delay_ms_;

    tf2_ros::StaticTransformBroadcaster static_broadcaster_;

    bool save_scan_;

    /// @brief 定位频率(定位间隔时间，多少秒1次)
    double loc_frequence_;

    /// @brief source点云最大点数量
    int maxpoints_source_ = 50000;
    /// @brief target点云最大点数量
    int maxpoints_target_ = 200000;

    /// @brief 初始化成功标志
    bool loc_initialized_ = false;

    /// @brief 当前定位overlap，confidence
    double loc_fitness_;

    /// @brief 定位置信度阈值
    double confidence_loc_th_;

    /// 卡尔曼滤波器
    KalmanFilter kf_baselink_x_;
    KalmanFilter kf_baselink_y_;
    KalmanFilter kf_baselink_z_;
    KalmanFilter kalman_filter_odom2map_;

    // 0:kf_processVar 1:kf_estimatedMeasVar
    std::vector<float> kf_param_x_;
    std::vector<float> kf_param_y_;
    std::vector<float> kf_param_z_;

    /// @brief 对odom2map进行kalman滤波
    bool filter_odom2map_ = false;
    double kalman_processVar2_ = 0.0;
    double kalman_estimatedMeasVar2_ = 0.0;

    /// 1202
    /// @brief 上次更新定位时的定位值
    Eigen::Vector3d last_loc_;
    // Eigen::Vector3d cur_loc_;
    /// @brief 更新地图子图的距离,超过则更新地图子图
    double dis_updatemap_;
};

GloabalLocalization::GloabalLocalization(ros::NodeHandle &nh, ros::NodeHandle &nh_private) : nh_(nh),
                                                                                             nh_private_(nh_private)
{
    flag_exit_ = false;
    loc_initialized_ = false;
    mat_baselink2odom_ = Eigen::Matrix4d::Identity();
    mat_odom2map_ = Eigen::Matrix4d::Identity();
    mat_initialpose_ = Eigen::Matrix4d::Identity();
    last_loc_ = Eigen::Vector3d(0, 0, -5000);

    pcd_map_ori_.reset(new open3d::geometry::PointCloud);
    pcd_map_coarse_.reset(new open3d::geometry::PointCloud);
    pcd_map_cur_.reset(new open3d::geometry::PointCloud);
    pcd_scan_cur_.reset(new open3d::geometry::PointCloud);
    pcd_map_fine_.reset(new open3d::geometry::PointCloud);
    queue_maxsize_ = 5;

    pub_baselink2map_ = nh.advertise<nav_msgs::Odometry>("/baselink2map", 100000);
    pub_baselink2map_kalman_ = nh.advertise<nav_msgs::Odometry>("/baselink2map_kalman", 100000);
    pub_motionlink2map_ = nh.advertise<nav_msgs::Odometry>("/motionlink2map", 100000);
    pub_odom2map_ = nh.advertise<nav_msgs::Odometry>("/odom2map", 100000);
    pub_odom2map_kalman_ = nh.advertise<nav_msgs::Odometry>("/odom2map_kalman", 100000);

    pub_map_ = nh.advertise<sensor_msgs::PointCloud2>("/3dmap", 1, true);
    pub_submap_ = nh.advertise<sensor_msgs::PointCloud2>("/submap", 1, true);
    pub_scan2map_ = nh.advertise<sensor_msgs::PointCloud2>("/scan2map", 1, true);
    pub_scan_ = nh.advertise<sensor_msgs::PointCloud2>("/scan", 1, true);
    pub_localization_3d_ = nh.advertise<geometry_msgs::PoseStamped>("/localization_3d", 1, false);
    // pub_localization_3d_ = nh.advertise<nav_msgs::Odometry>("/localization_3d", 1, false);
    pub_localization_3d_confidence_ = nh.advertise<std_msgs::Float32>("/localization_3d_confidence", 1, false);
    pub_localization_3d_delay_ms_ = nh.advertise<std_msgs::Float32>("/localization_3d_delay_ms", 1, false);

    loc_frequence_ = 2.0; //
    loc_fitness_ = 0.0;
    // 注册回调函数
    sub_baselink2odom_ = nh_.subscribe("/Odometry_loc", 50, &GloabalLocalization::CallbackBaselink2Odom, this);
    sub_scan_cur_ = nh_.subscribe("/cloud_registered_1", 50, &GloabalLocalization::CallbackScan, this);
    sub_initialpose_ = nh_.subscribe("/initialpose", 50, &GloabalLocalization::CallbackInitialPose, this);

    pose_baselink2odom_ = nav_msgs::Odometry();
    pose_baselink2odom_.header.frame_id = "odom";
    pose_baselink2odom_.child_frame_id = "base_link";
    // geometry_msgs的Quaternion会被初始化为0,0,0,0,而不是正确的0,0,0,1
    pose_baselink2odom_.pose.pose.orientation.w = 1;
    ROS_INFO("pose baselink2odom:\nx: %f, y: %f, z: %f, qx: %f, \
                            qy: %f, qz: %f, qw: %f",
             pose_baselink2odom_.pose.pose.position.x,
             pose_baselink2odom_.pose.pose.position.y,
             pose_baselink2odom_.pose.pose.position.z,
             pose_baselink2odom_.pose.pose.orientation.x,
             pose_baselink2odom_.pose.pose.orientation.y,
             pose_baselink2odom_.pose.pose.orientation.z,
             pose_baselink2odom_.pose.pose.orientation.w);

    // 队列最大数量
    nh_private_.param<int>("pcd_queue_maxsize", queue_maxsize_, 5);
    nh_private_.param<bool>("save_scan", save_scan_, false);
    /// 最大点数量限制
    nh_private_.param<int>("maxpoints_source", maxpoints_source_, 50000);
    nh_private_.param<int>("maxpoints_target", maxpoints_target_, 200000);

    // 定位间隔时间
    nh_private_.param<double>("loc_frequence", loc_frequence_, 2.0);

    /// 定位阈值
    nh_private_.param<double>("confidence_loc_th", confidence_loc_th_, 0.6);

    /// 卡尔曼参数
    nh_private_.param<std::vector<float>>("kf_baselink2map/x", kf_param_x_, std::vector<float>(2));
    nh_private_.param<std::vector<float>>("kf_baselink2map/y", kf_param_y_, std::vector<float>(2));
    nh_private_.param<std::vector<float>>("kf_baselink2map/z", kf_param_z_, std::vector<float>(2));

    nh_private_.param<bool>("filter_odom2map", filter_odom2map_, false);
    nh_private_.param<double>("kalman_processVar2", kalman_processVar2_, 0.02);
    nh_private_.param<double>("kalman_estimatedMeasVar2", kalman_estimatedMeasVar2_, 0.04);
    // voxelsize
    nh_private_.param<double>("voxelsize_coarse", voxelsize_coarse_, 0.2);
    nh_private_.param<double>("voxelsize_fine", voxelsize_fine_, 0.05);
    nh_private_.param<double>("threshold_fitness_init", threshold_fitness_init_, 0.9);
    nh_private_.param<double>("threshold_fitness", threshold_fitness_, 0.9);
    nh_private_.param<std::vector<float>>("initialpose", initialpose_, std::vector<float>());
    nh_private_.param<double>("dis_updatemap", dis_updatemap_, 5);

    for (auto i : initialpose_)
    {
        std::cout << i << " ";
    }
    std::cout << std::endl;
    mat_initialpose_.block<3, 3>(0, 0) = Euler2Matrix3d(Eigen::Vector3d(initialpose_[3], initialpose_[4], initialpose_[5]));
    mat_initialpose_.block<3, 1>(0, 3) = Eigen::Vector3d(initialpose_[0], initialpose_[1], initialpose_[2]);

    // 读取地图
    std::string path_map = "";
    nh_private_.param<std::string>("path_map", path_map, "");
    open3d::io::ReadPointCloud(path_map, *pcd_map_ori_);
    if (pcd_map_ori_ == nullptr || pcd_map_ori_->IsEmpty())
    {
        ROS_ERROR("read map from path: %s failed", path_map.c_str());
        ros::shutdown();
    }
    pcd_map_ori_->PaintUniformColor({1, 0, 0});

    pcd_map_coarse_ = pcd_map_ori_->VoxelDownSample(voxelsize_coarse_);
    pcd_map_coarse_->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxelsize_coarse_ * 2, 30));

    /// publish map, 用粗地图可视化，减少资源占用
    sensor_msgs::PointCloud2 pc2_map;
    open3d_conversions::open3dToRos(*pcd_map_coarse_, pc2_map);
    pc2_map.header.frame_id = "map";
    pc2_map.header.stamp = ros::Time::now();
    pub_map_.publish(pc2_map);

    pcd_map_fine_ = pcd_map_ori_->VoxelDownSample(voxelsize_fine_);
    pcd_map_fine_->EstimateNormals(open3d::geometry::KDTreeSearchParamHybrid(voxelsize_fine_ * 2, 30));


    GetTfTransformToMatrix("base_link", "imu_link", mat_imulink2baselink_);
    std::cout << "mat_imulink2baselink_:\n"
              << mat_imulink2baselink_ << std::endl;

    GetTfTransformToMatrix("motion_link", "base_link", mat_baselink2motionlink_);
    std::cout << "mat_baselink2motionlink_:\n"
              << mat_baselink2motionlink_ << std::endl;
    ROS_WARN("initialize finished");
}

GloabalLocalization::~GloabalLocalization()
{
    lock_exit_.lock();
    flag_exit_ = true;
    lock_exit_.unlock();
}

Eigen::Matrix3d GloabalLocalization::Euler2Matrix3d(const Eigen::Vector3d euler)
{
    Eigen::Matrix3d mat3d;
    // convert degrees to radians
    auto eulerAngle = euler / 180 * M_PI;
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle[0], Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle[1], Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle[2], Eigen::Vector3d::UnitZ()));
    mat3d = rollAngle * pitchAngle * yawAngle;
    return mat3d;
}
bool GloabalLocalization::GetTfTransformToMatrix(std::string frame_id, std::string child_frame_id, Eigen::Matrix4d &matrix)
{
    // 获取pose
    tf::StampedTransform pose_;
    tf::TransformListener tf_listener_;
    try
    {
        tf_listener_.waitForTransform(frame_id, child_frame_id, ros::Time(0), ros::Duration(3));
        tf_listener_.lookupTransform(frame_id, child_frame_id, ros::Time(0), pose_);
    }
    catch (tf::TransformException &e)
    {
        ROS_ERROR("[GetTransformMatrix]: %s", e.what());
        return false;
    }

    Eigen::Vector3d translation = Eigen::Vector3d(pose_.getOrigin().x(), pose_.getOrigin().y(), pose_.getOrigin().z());
    Eigen::Quaterniond quat = Eigen::Quaterniond::Identity();

    quat = Eigen::Quaterniond(pose_.getRotation().w(),
                              pose_.getRotation().x(),
                              pose_.getRotation().y(),
                              pose_.getRotation().z());
    Eigen::Matrix3d rotation = quat.matrix();

    matrix = Eigen::Matrix4d::Identity();
    matrix.block<3, 3>(0, 0) = rotation;
    matrix.matrix().block<3, 1>(0, 3) = translation;
    return true;
}

void GloabalLocalization::CallbackBaselink2Odom(
    const nav_msgs::Odometry::ConstPtr &baselink2odom)
{
    auto odom_cbk_s = std::chrono::high_resolution_clock::now();
    lock_timestamp_.lock();
    timestamp_odom_ = baselink2odom->header.stamp;
    lock_timestamp_.unlock();
    Eigen::Isometry3d mat_current = Eigen::Isometry3d::Identity();
    tf::poseMsgToEigen(baselink2odom->pose.pose, mat_current);
    auto mat_imulink2odom = mat_current.matrix();

    mat_baselink2odom_ = mat_imulink2odom * mat_imulink2baselink_.inverse();

    Eigen::Isometry3d Isometry3d_baselink2map;
    mat_baselink2map_ = mat_odom2map_ * mat_baselink2odom_;
    Isometry3d_baselink2map.matrix() = mat_baselink2map_;
    nav_msgs::Odometry baselink2map;
    tf::poseEigenToMsg(Isometry3d_baselink2map, baselink2map.pose.pose);
    baselink2map.header.frame_id = "map";
    baselink2map.child_frame_id = "base_link";
    baselink2map.header.stamp = baselink2odom->header.stamp;
    pub_baselink2map_.publish(baselink2map);

    Eigen::Isometry3d Isometry3d_odom2map;
    Isometry3d_odom2map.matrix() = mat_odom2map_;
    nav_msgs::Odometry odom2map;
    tf::poseEigenToMsg(Isometry3d_odom2map, odom2map.pose.pose);
    odom2map.header.frame_id = "map";
    odom2map.child_frame_id = "odom";
    odom2map.header.stamp = baselink2odom->header.stamp;
    pub_odom2map_.publish(odom2map);

    /// 发布tf关系
    static tf::TransformBroadcaster br_odom2map;
    tf::Transform transform_odom2map;
    tf::Quaternion q_odom2map;
    transform_odom2map.setOrigin(tf::Vector3(odom2map.pose.pose.position.x,
                                             odom2map.pose.pose.position.y,
                                             odom2map.pose.pose.position.z));
    q_odom2map.setW(odom2map.pose.pose.orientation.w);
    q_odom2map.setX(odom2map.pose.pose.orientation.x);
    q_odom2map.setY(odom2map.pose.pose.orientation.y);
    q_odom2map.setZ(odom2map.pose.pose.orientation.z);
    transform_odom2map.setRotation(q_odom2map);
    br_odom2map.sendTransform(tf::StampedTransform(transform_odom2map, baselink2odom->header.stamp, "map", "odom")); /// odom就是camera_init

    /// 卡尔曼滤波
    if (loc_initialized_)
    {
        Eigen::Matrix4d mat_baselink2map_kalman = Eigen::Matrix4d::Identity();

        if (filter_odom2map_)
        {
            Eigen::Isometry3d Isometry3d_odom2map_kalman;
            Isometry3d_odom2map_kalman.matrix() = mat_odom2map_kalman_;
            nav_msgs::Odometry odom2map_kalman;
            tf::poseEigenToMsg(Isometry3d_odom2map_kalman, odom2map_kalman.pose.pose);
            odom2map_kalman.header.frame_id = "map";
            odom2map_kalman.child_frame_id = "odom_kalman";
            odom2map_kalman.header.stamp = baselink2odom->header.stamp;
            pub_odom2map_kalman_.publish(odom2map_kalman);

            kf_baselink_z_.inputLatestNoisyMeasurement((mat_odom2map_kalman_ * mat_baselink2odom_)(2, 3));
            mat_baselink2map_kalman = mat_odom2map_kalman_ * mat_baselink2odom_;
        }
        else
        {
            kf_baselink_x_.inputLatestNoisyMeasurement((mat_baselink2map_)(0, 3));
            kf_baselink_y_.inputLatestNoisyMeasurement((mat_baselink2map_)(1, 3));
            kf_baselink_z_.inputLatestNoisyMeasurement((mat_baselink2map_)(2, 3));
            mat_baselink2map_kalman = mat_baselink2map_;
        }

        mat_baselink2map_kalman(2, 3) = kf_baselink_z_.getLatestEstimatedMeasurement();
        Eigen::Isometry3d Isometry3d_baselink2map_kalman;

        Isometry3d_baselink2map_kalman.matrix() = mat_baselink2map_kalman;
        nav_msgs::Odometry baselink2map_kalman;
        tf::poseEigenToMsg(Isometry3d_baselink2map_kalman, baselink2map_kalman.pose.pose);
        baselink2map_kalman.header.frame_id = "map";
        // baselink2map_kalman.child_frame_id = "base_link_kalman";
        baselink2map_kalman.header.stamp = baselink2odom->header.stamp;
        pub_baselink2map_kalman_.publish(baselink2map_kalman);

        Eigen::Matrix4d mat_motionlink2map = mat_baselink2map_kalman * mat_baselink2motionlink_.inverse();
        Eigen::Isometry3d Isometry3d_motionlink2map;
        Isometry3d_motionlink2map.matrix() = mat_motionlink2map;
        nav_msgs::Odometry motionlink2map;
        tf::poseEigenToMsg(Isometry3d_motionlink2map, motionlink2map.pose.pose);
        motionlink2map.header.frame_id = "map";
        // baselink2map_kalman.child_frame_id = "base_link_kalman";
        motionlink2map.header.stamp = baselink2odom->header.stamp;
        pub_motionlink2map_.publish(motionlink2map);

        /// 发布tf关系
        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(motionlink2map.pose.pose.position.x,
                                        motionlink2map.pose.pose.position.y,
                                        motionlink2map.pose.pose.position.z));
        q.setW(motionlink2map.pose.pose.orientation.w);
        q.setX(motionlink2map.pose.pose.orientation.x);
        q.setY(motionlink2map.pose.pose.orientation.y);
        q.setZ(motionlink2map.pose.pose.orientation.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, baselink2odom->header.stamp, "map", "motion_link"));

        localization_3d_confidence_.data = loc_fitness_;
        pub_localization_3d_confidence_.publish(localization_3d_confidence_);
        localization_3d_delay_ms_.data = (ros::Time::now().toSec() - baselink2odom->header.stamp.toSec()) * 1000.0;
        pub_localization_3d_delay_ms_.publish(localization_3d_delay_ms_);
        localization_3d_.header.frame_id = "map";
        localization_3d_.header.stamp = baselink2odom->header.stamp;
        localization_3d_.pose = motionlink2map.pose.pose;
        pub_localization_3d_.publish(localization_3d_);
    }
}
void GloabalLocalization::CallbackScan(
    const sensor_msgs::PointCloud2::ConstPtr &scan_in_baselink)
{
    auto cbk_s = std::chrono::high_resolution_clock::now();
    open3d::geometry::PointCloud pcd_recieved;
    // 单帧转换为open3d，几百us
    open3d_conversions::rosToOpen3d(scan_in_baselink, pcd_recieved);
    // 入队列
    // pcd_recieved
    if (que_pcd_scan_.size() >= queue_maxsize_)
    {
        std::queue<open3d::geometry::PointCloud> que_temp;
        lock_scan_.lock();
        pcd_scan_cur_->Clear();
        while (!que_pcd_scan_.empty())
        {
            *pcd_scan_cur_ += que_pcd_scan_.front();
            que_temp.push(que_pcd_scan_.front());
            que_pcd_scan_.pop();
        }
        lock_scan_.unlock();
        while (!que_temp.empty())
        {
            que_pcd_scan_.push(que_temp.front());
            que_temp.pop();
        }
        // 丢弃一个最旧的数据
        que_pcd_scan_.pop();
    }
    // 放入最新数据
    que_pcd_scan_.push(pcd_recieved);

    auto cbk_e = std::chrono::high_resolution_clock::now();
}

void GloabalLocalization::LocalizationInitialize()
{
    /// 裁剪后的地图
    std::shared_ptr<open3d::geometry::PointCloud> map_coarse_crop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> map_fine_crop(new open3d::geometry::PointCloud);

    /// 当前环境感知子图点云
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan(new open3d::geometry::PointCloud);
    /// 环境感知子图转换到地图坐标系
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan2map(new open3d::geometry::PointCloud);

    /// 用于配准的source target
    std::shared_ptr<open3d::geometry::PointCloud> source(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target(new open3d::geometry::PointCloud);

    /// cropbox,用于裁剪地图和当前环境感知子图
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_map(new open3d::geometry::OrientedBoundingBox);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_scan(new open3d::geometry::OrientedBoundingBox);

    /// 当前baselink到odom(camera_init)和map坐标系的关系
    Eigen::Matrix4d mat_baselink2odom_cur = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d mat_baselink2map_cur = Eigen::Matrix4d::Identity();

    /// 固定感知子图/历史地图子图大小
    OBB_map->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_map->color_ = Eigen::Vector3d(1, 0.5, 0);
    OBB_scan->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_scan->color_ = Eigen::Vector3d(0, 1, 0);

    double fitness_initial; /// overlap
    double loc_cost = 0;    /// 定位耗时(ms)
    int count_success = 0;
    while (1)
    {
        auto loc_s = std::chrono::high_resolution_clock::now(); /// 开始定位计时
        lock_scan_.lock();
        if (pcd_scan_cur_->IsEmpty())
        {
            lock_scan_.unlock();
            open3d::utility::LogInfo("wait for pcd_scan_cur_");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        else
        {
            /// 获取最新关系
            mat_baselink2odom_cur = mat_baselink2odom_;
            mat_baselink2map_cur = mat_baselink2map_;
            *pcd_scan = *pcd_scan_cur_;
            lock_scan_.unlock();
            lock_mat_odom2map_.lock();

            /// 将cropbox转换到对应位置进行裁剪点云
            OBB_map->center_ = mat_baselink2map_cur.block<3, 1>(0, 3);
            OBB_map->R_ = mat_baselink2map_cur.block<3, 3>(0, 0);
            OBB_scan->center_ = mat_baselink2odom_cur.block<3, 1>(0, 3);
            OBB_scan->R_ = mat_baselink2odom_cur.block<3, 3>(0, 0);
            *map_fine_crop = *pcd_map_fine_->Crop(*OBB_map);

            /// 配准计时
            auto reg0_s = std::chrono::high_resolution_clock::now();

            Eigen::Matrix4d reg_matrix = Eigen::Matrix4d::Identity();
            reg_matrix = mat_odom2map_;

            *target = *map_fine_crop;
            open3d::utility::LogInfo("before sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");
            if (target->points_.size() > maxpoints_target_)
            {
                target = target->RandomDownSample(double(maxpoints_target_) / target->points_.size());
            }
            open3d::utility::LogInfo("after sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");

            source = pcd_scan->Crop(*OBB_scan);
            open3d::utility::LogInfo("source size: {}, has normal: {}", source->points_.size(), source->HasNormals() ? "true" : "false");
            if (source->points_.size() > maxpoints_source_)
            {
                source = source->RandomDownSample(double(maxpoints_source_) / source->points_.size());
            }

            open3d::utility::LogInfo("source size: {}, has normal: {}", source->points_.size(), source->HasNormals() ? "true" : "false");

            source->Transform(reg_matrix);
            *pcd_scan2map = *source;

            auto multiScale_reg_matrix = pcd_tools::RegistrationMultiScaleIcp(source, target, voxelsize_fine_, 1, {1, 4, 6});
            reg_matrix = multiScale_reg_matrix * reg_matrix;
            source->Transform(multiScale_reg_matrix);
            auto eva_result_coarse = open3d::pipelines::registration::EvaluateRegistration(*source, *target, voxelsize_fine_ * 3);
            open3d::utility::LogInfo("eva fitness: {}", eva_result_coarse.fitness_);
            fitness_initial = eva_result_coarse.fitness_;
            *pcd_scan2map = *source;

            mat_odom2map_ = reg_matrix;
            lock_mat_odom2map_.unlock();
            auto loc_e = std::chrono::high_resolution_clock::now(); /// 结束定位计时
            loc_cost = std::chrono::duration_cast<std::chrono::microseconds>(loc_e - loc_s).count() / 1000.0;
            ROS_INFO("localization cost: %f ms", loc_cost);

            if (fitness_initial > threshold_fitness_init_)
            {
                count_success += 1;
                /// 连续两次定位成功后定位初始化成功
                if (count_success >= 2)
                {
                    break;
                }
            }
            else
            {
                count_success = 0;
            }
        }
    }


    open3d::utility::LogInfo("\n\n\nlocalization initialize success!!!!\n\n\n");
}
void GloabalLocalization::Localization()
{
    ROS_INFO("wait for Odometry_loc");
    ros::topic::waitForMessage<nav_msgs::Odometry>("/Odometry_loc");
    ROS_INFO("wait for cloud_registered_1");
    ros::topic::waitForMessage<sensor_msgs::PointCloud2>("/cloud_registered_1");
    // initialize
    /****初始化定位****/
    mat_odom2map_ = mat_initialpose_; /// 初始位姿，从目前是从配置文件给
    LocalizationInitialize();

    /// 卡尔曼滤波初始化
    kf_baselink_x_.KalmanFilterInit(kf_param_x_[0], kf_param_x_[1], mat_baselink2map_(0, 3), 1);
    kf_baselink_y_.KalmanFilterInit(kf_param_y_[0], kf_param_y_[1], mat_baselink2map_(1, 3), 1);
    kf_baselink_z_.KalmanFilterInit(kf_param_z_[0], kf_param_z_[1], mat_baselink2map_(2, 3), 1);
    kalman_filter_odom2map_.KalmanFilterInit(kalman_processVar2_, kalman_estimatedMeasVar2_, mat_baselink2map_(2, 3), 1);

    loc_initialized_ = true; /// 初始化成功

    double fitness = 0;
    auto coordinate_ori = open3d::geometry::TriangleMesh::CreateCoordinateFrame(2.0);
    auto coordinate_loc = open3d::geometry::TriangleMesh::CreateCoordinateFrame(2.0);
    auto coordinate_OBB_scan = open3d::geometry::TriangleMesh::CreateCoordinateFrame(2.0);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scancrop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_scan2map(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> source(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> target(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> map_coarse_crop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> map_fine_crop(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::PointCloud> pcd_submap(new open3d::geometry::PointCloud);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_map(new open3d::geometry::OrientedBoundingBox);
    std::shared_ptr<open3d::geometry::OrientedBoundingBox> OBB_scan(new open3d::geometry::OrientedBoundingBox);
    OBB_map->color_ = Eigen::Vector3d(1, 0.5, 0);
    OBB_map->extent_ = Eigen::Vector3d(60, 60, 40);

    OBB_scan->extent_ = Eigen::Vector3d(60, 60, 40);
    OBB_scan->color_ = Eigen::Vector3d(0, 1, 0);
    ros::Time time_current = timestamp_odom_;
    ros::Time time_last = time_current - ros::Duration(3.0);

    ROS_INFO("time_last: %f", time_last.toSec());
    ROS_INFO("time_current: %f", time_current.toSec());
    int scan_count = 0;

    std::string save_path = "/home/carlos/mount/E/lixin/data/yq_bag/scan_submap/";

    double time_diff_loc = 5;                                     /// 前后两次定位的时间差(s)
    std::chrono::high_resolution_clock::time_point time_last_loc; /// 上次定位的完成时间点
    std::chrono::high_resolution_clock::time_point time_this_loc; /// 当前定位的开始时间点
    double loc_cost = 0;                                          /// 定位耗时(ms)
    while (1)
    {

        lock_timestamp_.lock();
        time_current = timestamp_odom_;
        lock_timestamp_.unlock();
        auto time_diff_frame = time_current.toSec() - time_last.toSec();
        time_last = time_current;
        if (std::fabs(time_diff_frame) < 1e-6)
        {
            loc_cost = 0.0;
            continue;
        }

        time_this_loc = std::chrono::high_resolution_clock::now();
        time_diff_loc = std::chrono::duration_cast<std::chrono::microseconds>(time_this_loc - time_last_loc).count() / 1000000.0 + loc_cost / 1000.0;

        if (time_diff_loc < loc_frequence_)
        {
            int wait_time = int((loc_frequence_ - time_diff_loc) * 1000);
            open3d::utility::LogInfo("\n\ntime_this_loc: {}, time_last: {},\ntime_diff: {} s, sleep {} ms",
                                     std::chrono::duration_cast<std::chrono::milliseconds>(time_this_loc.time_since_epoch()).count() / 1000.0,
                                     std::chrono::duration_cast<std::chrono::milliseconds>(time_last_loc.time_since_epoch()).count() / 1000.0, time_diff_loc, wait_time);
            std::this_thread::sleep_for(std::chrono::milliseconds(wait_time));
        }
        else
        {
            open3d::utility::LogInfo("\n\ntime_diff:{} s, localization right now", time_diff_loc);
        }
        auto loc_s = std::chrono::high_resolution_clock::now(); /// 开始定位计时

        lock_scan_.lock();
        if (pcd_scan_cur_->IsEmpty())
        {
            lock_scan_.unlock();
            ROS_INFO("wait for pcd_scan_cur_");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            continue;
        }
        else
        {
            /// 是否对odom2map进行kalman滤波
            if (filter_odom2map_)
            {
                kalman_filter_odom2map_.inputLatestNoisyMeasurement(mat_odom2map_(2, 3));
                kalman_filter_odom2map_.inputLatestNoisyMeasurement(mat_odom2map_(2, 3)); /// 两次
                mat_odom2map_kalman_ = mat_odom2map_;
                mat_odom2map_kalman_(2, 3) = kalman_filter_odom2map_.getLatestEstimatedMeasurement();
            }
            Eigen::Matrix4d mat_baselink2odom_cur = Eigen::Matrix4d::Identity();
            Eigen::Matrix4d mat_baselink2map_cur = Eigen::Matrix4d::Identity();

            mat_baselink2odom_cur = mat_baselink2odom_;
            mat_baselink2map_cur = mat_baselink2map_;
            *pcd_scan = *pcd_scan_cur_;
            lock_scan_.unlock();
            Eigen::Vector3d cur_loc(mat_baselink2map_cur(0, 3), mat_baselink2map_cur(1, 3), mat_baselink2map_cur(2, 3));
            auto dis_motion = ComputeMotionDis(last_loc_, cur_loc);
            if (dis_motion > dis_updatemap_)
            {
                auto submap_s = std::chrono::high_resolution_clock::now();

                open3d::utility::LogInfo("\n***\n****\n***\n\n\nlast map update loc: x: {}, y: {}, z{},\n\
                now loc: x: {}, y: {}, z{}, 3d distance: {}, now needpdate submap",
                                         last_loc_.x(), last_loc_.y(), last_loc_.z(), cur_loc.x(), cur_loc.y(), cur_loc.z(), dis_motion);
                last_loc_ = cur_loc;
                OBB_map->center_ = mat_baselink2map_cur.block<3, 1>(0, 3);
                OBB_map->R_ = mat_baselink2map_cur.block<3, 3>(0, 0);

                /// 粗地图和精地图
                *map_fine_crop = *pcd_map_fine_->Crop(*OBB_map);

                auto submap_e = std::chrono::high_resolution_clock::now();
                auto submap_cost = std::chrono::duration_cast<std::chrono::microseconds>(submap_e - submap_s).count() / 1000.0;
                ROS_INFO("submap_cost: %f ms", submap_cost);
            }

            OBB_scan->center_ = mat_baselink2odom_cur.block<3, 1>(0, 3);
            OBB_scan->R_ = mat_baselink2odom_cur.block<3, 3>(0, 0);

            auto reg0_s = std::chrono::high_resolution_clock::now();

            Eigen::Matrix4d reg_matrix = Eigen::Matrix4d::Identity();

            lock_mat_odom2map_.lock();
            reg_matrix = mat_odom2map_;

            *target = *map_fine_crop;
            open3d::utility::LogInfo("before sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");
            if (target->points_.size() > maxpoints_target_)
            {
                target = target->RandomDownSample(double(maxpoints_target_) / target->points_.size());
            }
            open3d::utility::LogInfo("after sample, target size: {}, has normal: {}", target->points_.size(), target->HasNormals() ? "true" : "false");

            source = pcd_scan->Crop(*OBB_scan);
            open3d::utility::LogInfo("source size: {}, maxpoints_source_: {}", source->points_.size(), maxpoints_source_);
            source = source->VoxelDownSample(voxelsize_fine_);
            open3d::utility::LogInfo("source size after voxel downsample: {}", source->points_.size());
            if (source->points_.size() > maxpoints_source_)
            {
                source = source->RandomDownSample(double(maxpoints_source_) / source->points_.size());
            }
            open3d::utility::LogInfo("after prerpocess: {}", source->points_.size());

            auto reg_result2 = pcd_tools::RegistrationIcp(source, target, voxelsize_fine_ * 2, reg_matrix, 1);
            reg_matrix = reg_result2.transformation_ * reg_matrix;
            auto eva_result2 = open3d::pipelines::registration::EvaluateRegistration(*source, *target, voxelsize_fine_ * 4, reg_matrix);
            /// 给发布的置信度赋值
            loc_fitness_ = eva_result2.fitness_;
            open3d::utility::LogInfo("reg_result.fitness: {}, eva fitness: {}", reg_result2.fitness_, eva_result2.fitness_);
            /// 超过阈值才更新,防止因配准结果有问题而导致定位出问题
            if (loc_fitness_ > threshold_fitness_)
            {
                mat_odom2map_ = reg_matrix;
            }
            lock_mat_odom2map_.unlock();

            // save_path
            if (save_scan_)
            {
                pcd_scan->Transform(mat_baselink2odom_cur.inverse());
                pcd_scan2map->Transform(mat_baselink2map_cur.inverse());
                open3d::io::WritePointCloud(save_path + std::to_string(scan_count) + "_ori.ply", *pcd_scan);
                open3d::io::WritePointCloud(save_path + std::to_string(scan_count) + "_crop.ply", *pcd_scan2map);
                scan_count += 1;
            }

            auto loc_e = std::chrono::high_resolution_clock::now(); /// 结束定位计时
            time_last_loc = loc_e;
            loc_cost = std::chrono::duration_cast<std::chrono::microseconds>(loc_e - loc_s).count() / 1000.0;
            ROS_INFO("localization cost: %f ms", loc_cost);
        }
    }
}

void GloabalLocalization::StartLoc()
{
    thread_loc_ = std::thread(&GloabalLocalization::Localization, this);
}

void GloabalLocalization::CallbackInitialPose(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &initialpose)
{
    std::cout << "mat_odom2map_\n"
              << mat_odom2map_ << std::endl;
    std::cout << "confidence_loc_th_: " << confidence_loc_th_ << " current confidence: " << loc_fitness_ << std::endl;

    if (!(loc_initialized_ && loc_fitness_ > 0.99))
    {
        std::cout << "initpose:x y z, x y z w\n"
                  << initialpose->pose.pose.position.x << " "
                  << initialpose->pose.pose.position.y << " "
                  << initialpose->pose.pose.position.z << " "
                  << initialpose->pose.pose.orientation.x << " "
                  << initialpose->pose.pose.orientation.y << " "
                  << initialpose->pose.pose.orientation.z << " "
                  << initialpose->pose.pose.orientation.w << std::endl;

        Eigen::Quaterniond rotation_q;
        rotation_q.w() = initialpose->pose.pose.orientation.w;
        rotation_q.x() = initialpose->pose.pose.orientation.x;
        rotation_q.y() = initialpose->pose.pose.orientation.y;
        rotation_q.z() = initialpose->pose.pose.orientation.z;
        mat_initialpose_.block<3, 3>(0, 0) = rotation_q.matrix();
        mat_initialpose_.block<3, 1>(0, 3) = Eigen::Vector3d(initialpose->pose.pose.position.x, initialpose->pose.pose.position.y, initialpose->pose.pose.position.z);
        lock_mat_odom2map_.lock();
        mat_odom2map_ = mat_initialpose_;
        lock_mat_odom2map_.unlock();
        std::cout << "\n\n*** update mat_odom2map_" << std::endl;
    }
    std::cout << "mat_odom2map_\n"
              << mat_odom2map_ << std::endl;
}
double GloabalLocalization::ComputeMotionDis(const Eigen::Vector3d &a, const Eigen::Vector3d &b)
{
    return std::sqrt(std::pow(a.x() - b.x(), 2) + std::pow(a.y() - b.y(), 2) + std::pow(a.z() - b.z(), 2));
}

int main(int argc, char *argv[])
{
    ros::init(argc, argv, "global_loc_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    // global_odom.
    std::cout << "start spin" << std::endl;
    // 创建异步对象
    ros::AsyncSpinner spinner(3);
    // 开始异步处理
    spinner.start();

    GloabalLocalization global_loc(nh, nh_private);
    global_loc.StartLoc();

    // 等待节点关闭
    ros::waitForShutdown();

    return 0;
}