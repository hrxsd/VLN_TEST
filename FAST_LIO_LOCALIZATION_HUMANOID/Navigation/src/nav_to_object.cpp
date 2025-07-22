#include <ros/ros.h>
#include <ros/service_client.h>
#include <yaml-cpp/yaml.h>
#include <actionlib/client/simple_action_client.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <std_msgs/String.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PoseStamped.h>
#include <math.h>
#include <nav_msgs/GetPlan.h>
#include <nav_msgs/Path.h>
#include <fstream>
#include <nav_msgs/OccupancyGrid.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>


class NavigationByName 
{
public:
    NavigationByName() 
    {
        // 加载yaml文件
        semantic_info_file = "/home/nuc1003a/ARTS_TEST/src/hdl_localization/config/map/semantic_map.yaml";
        load_semantic_info();

        // init move_base
        client = new actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction>("move_base", true);
        client->waitForServer();

        object_name_subscriber = nh.subscribe("/object_name", 10, &NavigationByName::find_object_by_name, this);
        robot_pose_subscriber = nh.subscribe("/robot_pose_ekf/odom_combined", 10, &NavigationByName::update_robot_pose, this);
        object_id_subscriber = nh.subscribe("/object_id", 10, &NavigationByName::navigate_to_object, this);
        global_costmap_subscriber = nh.subscribe("/move_base/global_costmap/costmap", 1, &NavigationByName::global_costmap_callback, this);
    }

    ~NavigationByName() 
    {
        delete client;
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber object_name_subscriber;
    ros::Subscriber robot_pose_subscriber;
    ros::Subscriber object_id_subscriber;
    ros::Subscriber global_costmap_subscriber;
    actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction>* client;
    std::string semantic_info_file;
    std::vector<YAML::Node> semantic_info;
    geometry_msgs::Pose robot_pose;
    bool new_gaol_received = false;
    nav_msgs::OccupancyGrid global_costmap;

    // 加载yaml文件的语义信息
    void load_semantic_info() 
    {
        std::ifstream file(semantic_info_file);
        YAML::Node data = YAML::Load(file);
        semantic_info = data["semantic_info"].as<std::vector<YAML::Node>>();
    }

    void update_robot_pose(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& msg) 
    {
        robot_pose = msg->pose.pose;
    }

    // 根据物体名称查找物体信息
    void find_object_by_name(const std_msgs::String::ConstPtr& msg) 
    {
        std::string object_name = msg->data;
        ROS_INFO("Received request to find objects with name: %s", object_name.c_str());
        std::vector<YAML::Node> matching_objects = get_semantic_info_by_name(object_name);
        if (!matching_objects.empty()) 
        {
            ROS_INFO("Found %lu objects with name '%s'", matching_objects.size(), object_name.c_str());
            // for (const auto& obj : matching_objects) 
            // {
            //     ROS_INFO("Object ID '%d', Position: (%f, %f)", obj["id"].as<int>(), obj["position"]["x"].as<double>(), obj["position"]["y"].as<double>());
            // }
        } 
        else 
        {
            ROS_WARN("No objects found with name '%s'", object_name.c_str());
        }
    }

    // 
    std::vector<YAML::Node> get_semantic_info_by_name(const std::string& name) 
    {
        std::vector<YAML::Node> matching_objects;
        for (const auto& semantic : semantic_info) 
        {
            if (semantic["name"].as<std::string>() == name) 
            {
                matching_objects.push_back(semantic);
            }
        }
        return matching_objects;
    }


    YAML::Node get_semantic_info_by_id(int object_id) 
    {
        for (const auto& semantic : semantic_info) 
        {
            if (semantic["id"].as<int>() == object_id) 
            {
                return semantic;
            }
        }
        return YAML::Node();
    }

    void global_costmap_callback(const nav_msgs::OccupancyGrid::ConstPtr& msg) 
    {
        // ROS_INFO("Received global costmap");
        global_costmap = *msg;
        ROS_INFO("Received global costmap: width=%d, height=%d", global_costmap.info.width, global_costmap.info.height);
    }

    void navigate_to_object(const std_msgs::String::ConstPtr& msg) 
    {
        ros::Time start_time = ros::Time::now();

        int object_id = std::stoi(msg->data);
        ROS_INFO("Received request to navigate to the object with ID: %d", object_id);

        // 获取目标物体的语义信息
        YAML::Node semantic_info = get_semantic_info_by_id(object_id);

        if (semantic_info) 
        {
            YAML::Node position = semantic_info["position"];
            YAML::Node size = semantic_info["size"].IsDefined() ? semantic_info["size"] : YAML::Node(YAML::NodeType::Map);
            if (!semantic_info["size"].IsDefined()) 
            {
                size["x"] = 0.5;
                size["y"] = 0.5;
            }
            ROS_INFO("Detected objected ID %d ", object_id);
            double safe_distance = 0.9;  // Define a safe distance in meters
            if (!robot_pose.position.x == 0.0 && !robot_pose.position.y == 0.0) 
            {
                geometry_msgs::Point adjusted_position;
                geometry_msgs::Quaternion orientation;

                ros::Time move_base_start_time = ros::Time::now();
                calculate_safe_position_and_orientation(position, size, safe_distance, adjusted_position, orientation);

                // ROS_INFO("取消当前目标。。。");
                new_gaol_received = true;
                client->cancelAllGoals();
                
                // ros::Time move_base_start_time = ros::Time::now();
                send_goal(adjusted_position, orientation);
                ros::Time move_base_end_time = ros::Time::now();
                ros::Duration move_base_duration = move_base_end_time - move_base_start_time;
                ROS_INFO("Time taken to send goal to move_base: %f seconds", move_base_duration.toSec());
            } 
            else 
            {
                ROS_WARN("Robot's current position is not available.");
            }
        } 
        else 
        {
            ROS_WARN("Object with ID '%d' not found in semantic info.", object_id);
        }

    }

    void calculate_safe_position_and_orientation(const YAML::Node& position, const YAML::Node& size, double safe_distance, geometry_msgs::Point& adjusted_position, geometry_msgs::Quaternion& orientation)
    {
        double center_x = position["x"].as<double>();
        double center_y = position["y"].as<double>();

        int map_width = global_costmap.info.width;
        int map_height = global_costmap.info.height;
        double map_resolution = global_costmap.info.resolution;

        double robot_x = robot_pose.position.x;
        double robot_y = robot_pose.position.y;

        // 将物体中心坐标转换为地图索引
        int center_index_x = static_cast<int>((center_x - global_costmap.info.origin.position.x) / map_resolution);
        int center_index_y = static_cast<int>((center_y - global_costmap.info.origin.position.y) / map_resolution);

        ROS_INFO("Object center: (%f, %f), map index: (%d, %d)", center_x, center_y, center_index_x, center_index_y);

        if (center_index_x < 0 || center_index_x >= map_width || center_index_y < 0 || center_index_y >= map_height)
        {
            ROS_WARN("Object center out of map bounds");
            return;
        }

        double max_score = -std::numeric_limits<double>::max();
        geometry_msgs::Point best_position;

        // 遍历物体边缘附近的点
        for (double angle = 0; angle < 2 * M_PI; angle += M_PI / 36)
        {
            double x = center_x + (size["x"].as<double>() / 2 + safe_distance) * cos(angle);
            double y = center_y + (size["y"].as<double>() / 2 + safe_distance) * sin(angle);

            double gradient = calculate_gradient(center_x, center_y, x, y);
            double distance = hypot(x - robot_x, y - robot_y);

            double gradient_weight = 1.5;
            double distance_weight = 1.5;

            // double score = gradient_weight * gradient + distance_weight * distance;
            double score = gradient / distance + gradient_weight * gradient;

            ROS_INFO("Checked point (%f, %f), gradient: %f, distance: %f, score: %f", x, y, gradient, distance, score);

            if (score >= 0 && score > max_score)
            {
                max_score = score;
                best_position.x = x;
                best_position.y = y;
            }
        }

        if (max_score == std::numeric_limits<double>::max())
        {
            ROS_WARN("No valid target point found around the object");
            return;
        }

        adjusted_position = best_position;
        adjusted_position.z = 0.0;

        // 设置默认方向
        double facing_angle = atan2(position["y"].as<double>() - adjusted_position.y, position["x"].as<double>() - adjusted_position.x);
        orientation.z = sin(facing_angle / 2.0);
        orientation.w = cos(facing_angle / 2.0);

        ROS_INFO("Calculated safe position: (%f, %f)", adjusted_position.x, adjusted_position.y);
    }

    double calculate_gradient(double center_x, double center_y, double x, double y)
    {
        int map_width = global_costmap.info.width;
        int map_height = global_costmap.info.height;
        double map_resolution = global_costmap.info.resolution;

        // 将世界坐标转换为地图索引
        int center_index_x = static_cast<int>((center_x - global_costmap.info.origin.position.x) / map_resolution);
        int center_index_y = static_cast<int>((center_y - global_costmap.info.origin.position.y) / map_resolution);
        int point_index_x = static_cast<int>((x - global_costmap.info.origin.position.x) / map_resolution);
        int point_index_y = static_cast<int>((y - global_costmap.info.origin.position.y) / map_resolution);

        ROS_INFO("center_x: %f, center_y: %f, x: %f, y: %f", center_x, center_y, x, y);
        ROS_INFO("center_index_x: %d, center_index_y: %d, point_index_x: %d, point_index_y: %d", center_index_x, center_index_y, point_index_x, point_index_y);

        // 边界检查
        if (center_index_x < 0 || center_index_x >= map_width || center_index_y < 0 || center_index_y >= map_height ||
            point_index_x < 0 || point_index_x >= map_width || point_index_y < 0 || point_index_y >= map_height)
        {
            ROS_WARN("Point (%f, %f) or center (%f, %f) out of map bounds", x, y, center_x, center_y);
            return -1; // 返回一个负值表示无效的梯度
        }

        int center_index = center_index_y * map_width + center_index_x;
        int point_index = point_index_y * map_width + point_index_x;

        double center_cost = global_costmap.data[center_index];
        double point_cost = global_costmap.data[point_index];

        double distance = sqrt(pow(center_x - x, 2) + pow(center_y - y, 2));
        double gradient = (center_cost - point_cost) / distance;

        return gradient;
    }

    void send_goal(const geometry_msgs::Point& position, const geometry_msgs::Quaternion& orientation) 
    {
        move_base_msgs::MoveBaseGoal goal;
        goal.target_pose.header.frame_id = "map";
        goal.target_pose.header.stamp = ros::Time::now();
        goal.target_pose.pose.position.x = position.x;
        goal.target_pose.pose.position.y = position.y;
        goal.target_pose.pose.orientation = orientation;
        client->sendGoal(goal,
                         boost::bind(&NavigationByName::done_callback, this, _1, _2),
                         boost::bind(&NavigationByName::active_callback, this),
                         boost::bind(&NavigationByName::feedback_callback, this, _1));
        ROS_INFO("Sent goal to position: (%f, %f)", position.x, position.y);

    }

    void done_callback(const actionlib::SimpleClientGoalState& state, const move_base_msgs::MoveBaseResultConstPtr& result) 
    {
        if (state == actionlib::SimpleClientGoalState::SUCCEEDED) 
        {
            ROS_INFO("The robot has reached the goal!");
        } 
        else 
        {
            ROS_ERROR("The robot failed to reach goal. State: %s", state.toString().c_str());
        }
    }

    void active_callback()
    {
        // ROS_INFO("Goal just went active");
    
    }

    void feedback_callback(const move_base_msgs::MoveBaseFeedback::ConstPtr& feedback)
    {
        // ROS_INFO("Received feedback: current position (%f, %f)", feedback->base_position.pose.position.x, feedback->base_position.pose.position.y);
    }

};

int main(int argc, char** argv) {
    ros::init(argc, argv, "navigation_by_name");
    NavigationByName navigation_by_name;

    // 设置缓存时间为 20 秒
    tf2_ros::Buffer tfBuffer(ros::Duration(30.0));
    tf2_ros::TransformListener tfListener(tfBuffer);

    ros::spin();
    return 0;
}

    