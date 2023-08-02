#include <nav_msgs/Odometry.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl_conversions/pcl_conversions.h>

#include "include/STDesc.h"

// Read KITTI data
std::vector<float> read_lidar_data(const std::string lidar_data_path) {
    std::ifstream lidar_data_file;
    lidar_data_file.open(lidar_data_path,
                         std::ifstream::in | std::ifstream::binary);
    if (!lidar_data_file) {
        std::cout << "Read End..." << std::endl;
        std::vector<float> nan_data;
        return nan_data;
        // exit(-1);
    }
    lidar_data_file.seekg(0, std::ios::end);
    const size_t num_elements = lidar_data_file.tellg() / sizeof(float);
    lidar_data_file.seekg(0, std::ios::beg);

    std::vector<float> lidar_data_buffer(num_elements);
    lidar_data_file.read(reinterpret_cast<char *>(&lidar_data_buffer[0]),
                         num_elements * sizeof(float));
    return lidar_data_buffer;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "demo_ntu");
    ros::NodeHandle nh;
    std::string generate_lidar_path = "";
    std::string generate_pose_path = "";
    std::string localization_lidar_path = "";
    std::string localization_pose_path = "";
    std::string config_path = "";

    std::string descriptor_path = "";
    bool generateDataBase;

    std::string reference_gt_path = "";
    std::string global_map_path = "";

    nh.param<std::string>("generate_lidar_path", generate_lidar_path, "");
    nh.param<std::string>("generate_pose_path", generate_pose_path, "");
    nh.param<std::string>("localization_lidar_path", localization_lidar_path,
                          "");
    nh.param<std::string>("localization_pose_path", localization_pose_path, "");

    nh.param<bool>("generate_descriptor_database", generateDataBase, false);
    nh.param<std::string>("descriptor_file_path", descriptor_path, "");

    nh.param<std::string>("reference_gt_path", reference_gt_path, "");
    nh.param<std::string>("global_map_path", global_map_path, "");

    ConfigSetting config_setting;
    read_parameters(nh, config_setting);

    ros::Publisher pubOdomAftMapped =
            nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
    ros::Publisher pubRegisterCloud =
            nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
    ros::Publisher pubCureentCloud =
            nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
    ros::Publisher pubCurrentCorner =
            nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
    ros::Publisher pubMatchedCloud =
            nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
    ros::Publisher pubMatchedCorner = nh.advertise<sensor_msgs::PointCloud2>(
            "/cloud_matched_key_points", 100);
    ros::Publisher pubSTD = nh.advertise<visualization_msgs::MarkerArray>(
            "descriptor_line", 10);

    ros::Publisher pubMapCloud =
            nh.advertise<sensor_msgs::PointCloud2>("/map_cloud", 100);

    ros::Rate loop(500);
    ros::Rate slow_loop(10);

    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> generate_poses_vec;
    std::vector<std::string> generate_times_vec;
    std::vector<int> generate_index_vec;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>>
            generate_key_poses_vec;

    load_CSV_pose_with_time(generate_pose_path, generate_index_vec,
                            generate_poses_vec, generate_times_vec);

    std::cout << "Sucessfully load pose with number: "
              << generate_poses_vec.size() << std::endl;

    // load global map to visualize
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr globalMapPtr(
            new pcl::PointCloud<pcl::PointXYZRGB>);
    if (pcl::io::loadPCDFile<pcl::PointXYZRGB>(global_map_path,
                                               *globalMapPtr) == -1) {
        LOG(ERROR) << "Load Global ERROR!";
    } else {
        LOG(INFO) << "Global Map loaded...";
    }

    // voxdownsampling the global for only once
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr map_filtered(
            new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::VoxelGrid<pcl::PointXYZRGB> map_filter;
    map_filter.setLeafSize(0.25, 0.25, 0.25);
    map_filter.setInputCloud(globalMapPtr);
    map_filter.filter(*map_filtered);

    STDescManager *std_manager = new STDescManager(config_setting);

    size_t gen_total_size = generate_poses_vec.size();
    size_t cloudInd = 0;
    size_t keyCloudInd = 0;
    pcl::PointCloud<pcl::PointXYZI>::Ptr temp_cloud(
            new pcl::PointCloud<pcl::PointXYZI>());

    std::vector<double> descriptor_time;
    std::vector<double> querying_time;
    std::vector<double> update_time;
    std::vector<double> t_error_vec;
    std::vector<double> r_error_vec;
    int triggle_loop_num = 0;

    if (generateDataBase) {
        // generate descriptor database
        ROS_INFO("Generating descriptors ...");
        for (int cloudInd = 0; cloudInd < gen_total_size; ++cloudInd) {
            std::string ori_time_str = generate_times_vec[cloudInd];
            std::replace(ori_time_str.begin(), ori_time_str.end(), '.', '_');
            std::string curr_lidar_path = generate_lidar_path + "cloud_" +
                                          std::to_string(cloudInd + 1) + "_" +
                                          ori_time_str + ".pcd";

            pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(
                    new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud(
                    new pcl::PointCloud<pcl::PointXYZI>());
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_registered(
                    new pcl::PointCloud<pcl::PointXYZI>());

            Eigen::Vector3d translation = generate_poses_vec[cloudInd].first;
            Eigen::Matrix3d rotation = generate_poses_vec[cloudInd].second;
            if (pcl::io::loadPCDFile<pcl::PointXYZI>(curr_lidar_path,
                                                     *current_cloud) == -1) {
                ROS_ERROR("Couldn't read scan from file. \n");
                std::cout << "Current File Name is: "
                          << generate_index_vec[cloudInd] << std::endl;
                return (-1);
            }

            Eigen::Matrix4d curr_trans_matrix;
            curr_trans_matrix.setIdentity();
            curr_trans_matrix.topLeftCorner(3, 3) = rotation;
            curr_trans_matrix.topRightCorner(3, 1) = translation;
            pcl::transformPointCloud<pcl::PointXYZI>(
                    *current_cloud, *current_cloud, curr_trans_matrix);

            down_sampling_voxel(*current_cloud, config_setting.ds_size_);
            for (auto pv : current_cloud->points) {
                temp_cloud->points.push_back(pv);
            }

            if (cloudInd % config_setting.sub_frame_num_ == 0) {
                // step1. Descriptor Extraction
                // auto t_descriptor_begin =
                // std::chrono::high_resolution_clock::now();
                std::vector<STDesc> stds_vec;
                std_manager->GenerateSTDescs(temp_cloud, stds_vec);

                // step3. Add descriptors to the database
                // auto t_map_update_begin =
                // std::chrono::high_resolution_clock::now();
                std_manager->AddSTDescs(stds_vec);
                // auto t_map_update_end =
                // std::chrono::high_resolution_clock::now();
                // update_time.push_back(
                //     time_inc(t_map_update_end, t_map_update_begin));

                // pcl::PointCloud<pcl::PointXYZI> save_key_cloud;
                // save_key_cloud = *temp_cloud;

                // std_manager->key_cloud_vec_.push_back(
                //     save_key_cloud.makeShared());
                // generate_key_poses_vec.push_back(generate_poses_vec[cloudInd]);
            }

            if (cloudInd % 100 == 0) {
                ROS_INFO("Generated %d frames", cloudInd);
            }
            temp_cloud->clear();
        }

        // save generated things
        std_manager->saveToFile(descriptor_path);

        ROS_INFO("Generation done. Exiting program ... ");
        return 0;
    }

    // load STD descriptors in storage
    std_manager->loadExistingSTD(descriptor_path, generate_poses_vec.size());
    ROS_INFO("Loaded saved STD.");

    /////////////// localization //////////////
    bool flagStop = false;

    cloudInd = 0;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>>
            localization_poses_vec;
    std::vector<double> key_frame_times_vec;
    std::vector<int> localization_index_vec;
    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>>
            localization_key_poses_vec;
    load_keyframes_pose_pcd(localization_pose_path, localization_index_vec,
                            localization_poses_vec, key_frame_times_vec);

    std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>>
            reference_gt_pose_vec;
    std::vector<std::string> reference_time_vec_string;
    std::vector<double> reference_time_vec;
    std::vector<int> reference_index_vec;
    load_CSV_pose_with_time(reference_gt_path, reference_index_vec,
                            reference_gt_pose_vec, reference_time_vec_string);

    for (auto itr : reference_time_vec_string) {
        double temp = std::stod(itr);
        reference_time_vec.push_back(temp);
    }

    std::vector<int> resulted_index =
            findCorrespondingFrames(key_frame_times_vec, reference_time_vec);

    if (resulted_index.size() != key_frame_times_vec.size()) {
        ROS_ERROR("SIZE NOT MATCH !");
    }

    std::cout << "Sucessfully load data to be relocalized: "
              << localization_poses_vec.size() << std::endl;

    size_t loc_total_size = localization_poses_vec.size();

    ///////////// start retrieval /////////////////
    while (ros::ok()) {
        if (cloudInd >= loc_total_size) {
            break;
        }

        int curr_index = localization_index_vec[cloudInd];
        std::stringstream curr_lidar_path;
        curr_lidar_path << localization_lidar_path << "KfCloudinW_"
                        << std::setfill('0') << std::setw(3) << curr_index
                        << ".pcd";

        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(
                new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr gt_cloud(
                new pcl::PointCloud<pcl::PointXYZI>());
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_registered(
                new pcl::PointCloud<pcl::PointXYZI>());

        Eigen::Vector3d translation = localization_poses_vec[cloudInd].first;
        Eigen::Matrix3d rotation = localization_poses_vec[cloudInd].second;
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(curr_lidar_path.str(),
                                                 *current_cloud) == -1) {
            ROS_ERROR("Couldn't read scan from file. \n");
            std::cout << "Current File Name is: "
                      << localization_index_vec[cloudInd] << std::endl;
            return (-1);
        }

        Eigen::Matrix4d curr_trans_matrix;
        curr_trans_matrix.setIdentity();
        curr_trans_matrix.topLeftCorner(3, 3) = rotation;
        curr_trans_matrix.topRightCorner(3, 1) = translation;
        pcl::transformPointCloud<pcl::PointXYZI>(
                *current_cloud, *current_cloud,
                curr_trans_matrix.cast<float>().inverse());

        down_sampling_voxel(*current_cloud, config_setting.ds_size_);
        for (auto pv : current_cloud->points) {
            temp_cloud->points.push_back(pv);
        }

        // check if keyframe
        if (cloudInd % config_setting.sub_frame_num_ == 0) {
            std::cout << "Key Frame id:" << keyCloudInd
                      << ", cloud size: " << temp_cloud->size() << std::endl;
            // step1. Descriptor Extraction
            auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
            std::vector<STDesc> stds_vec;
            std_manager->GenerateSTDescsOneTime(temp_cloud, stds_vec);
            auto t_descriptor_end = std::chrono::high_resolution_clock::now();
            descriptor_time.push_back(
                    time_inc(t_descriptor_end, t_descriptor_begin));
            // step2. Searching Loop
            auto t_query_begin = std::chrono::high_resolution_clock::now();
            std::pair<int, double> search_result(-1, 0);
            std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
            loop_transform.first << 0, 0, 0;
            loop_transform.second = Eigen::Matrix3d::Identity();
            std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
            std_manager->SearchLoop(stds_vec, search_result, loop_transform,
                                    loop_std_pair);
            if (search_result.first > 0) {
                std::cout << "[Loop Detection] triggle loop: " << keyCloudInd
                          << "--" << search_result.first
                          << ", score:" << search_result.second << std::endl;

                // Compute Pose Estimation Error
                int match_frame = search_result.first;
                std_manager->PlaneGeomrtricIcp(
                        std_manager->current_plane_cloud_,
                        std_manager->plane_cloud_vec_[match_frame],
                        loop_transform);

                Eigen::Matrix4d estimated_transform;
                estimated_transform.topLeftCorner(3, 3) = loop_transform.second;
                estimated_transform.topRightCorner(3, 1) = loop_transform.first;

                pcl::transformPointCloud<pcl::PointXYZI>(
                        *temp_cloud, *cloud_registered,
                        (estimated_transform).cast<float>());

                Eigen::Matrix4d odom_trans_matrix;
                odom_trans_matrix.setIdentity();
                odom_trans_matrix.topLeftCorner(3, 3) = rotation;
                odom_trans_matrix.topRightCorner(3, 1) = translation;

                Eigen::Matrix4d estimated_pose_inW;
                estimated_pose_inW = estimated_transform;

                Eigen::Vector3d gt_translation =
                        reference_gt_pose_vec[resulted_index[keyCloudInd]]
                                .first;
                Eigen::Matrix3d gt_rotation =
                        reference_gt_pose_vec[resulted_index[keyCloudInd]]
                                .second;

                double t_e = (gt_translation -
                              estimated_pose_inW.topRightCorner(3, 1))
                                     .norm();
                double r_e =
                        std::abs(std::acos(fmin(
                                fmax(((gt_rotation *
                                       estimated_pose_inW.topLeftCorner(3, 3)
                                               .inverse())
                                              .trace() -
                                      1) / 2,
                                     -1.0),
                                1.0))) /
                        M_PI * 180;
                std::cout << "Estimated Translation Error is:  " << t_e << " m;"
                          << std::endl;
                std::cout << "Estimated Rotation Error is:     " << r_e
                          << " degree;" << std::endl;
                t_error_vec.push_back(t_e);
                r_error_vec.push_back(r_e);
                if (r_e > 100.0) {
                    flagStop = true;
                }
            }
            auto t_query_end = std::chrono::high_resolution_clock::now();
            querying_time.push_back(time_inc(t_query_end, t_query_begin));

            // step3. Add descriptors to the database
            // auto t_map_update_begin =
            // std::chrono::high_resolution_clock::now();
            // std_manager->AddSTDescs(stds_vec);
            // auto t_map_update_end =
            // std::chrono::high_resolution_clock::now(); update_time.push_back(
            //     time_inc(t_map_update_end, t_map_update_begin));
            std::cout << "[Time] descriptor extraction: "
                      << time_inc(t_descriptor_end, t_descriptor_begin)
                      << "ms, "
                      << "query: " << time_inc(t_query_end, t_query_begin)
                      << "ms, "
                      << "update map:"
                      << " Nan "
                      << "ms" << std::endl;
            std::cout << std::endl;

            // publish
            sensor_msgs::PointCloud2 pub_cloud;
            pcl::toROSMsg(*temp_cloud, pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCureentCloud.publish(pub_cloud);
            pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubCurrentCorner.publish(pub_cloud);

            // publish map for visualization
            sensor_msgs::PointCloud2 cloud_ROS;
            pcl::toROSMsg(*map_filtered, cloud_ROS);
            cloud_ROS.header.frame_id = "camera_init";
            pubMapCloud.publish(cloud_ROS);

            if (search_result.first >= 0) {
                triggle_loop_num++;
                pcl::toROSMsg(
                        *std_manager->plane_cloud_vec_[search_result.first],
                        pub_cloud);
                pub_cloud.header.frame_id = "camera_init";
                pubMatchedCloud.publish(pub_cloud);
                slow_loop.sleep();

                pcl::toROSMsg(*cloud_registered, pub_cloud);
                pub_cloud.header.frame_id = "camera_init";
                pubRegisterCloud.publish(pub_cloud);
                slow_loop.sleep();

                // pcl::toROSMsg(
                //     *std_manager->corner_cloud_vec_[search_result.first],
                //     pub_cloud);
                // pub_cloud.header.frame_id = "camera_init";
                // pubMatchedCorner.publish(pub_cloud);
                // publish_std_pairs(loop_std_pair, pubSTD);
                // slow_loop.sleep();
                if (flagStop) {
                    getchar();
                    flagStop = false;
                }
            }
            temp_cloud->clear();
            keyCloudInd++;
            slow_loop.sleep();
        }
        nav_msgs::Odometry odom;
        odom.header.frame_id = "camera_init";
        odom.pose.pose.position.x = translation[0];
        odom.pose.pose.position.y = translation[1];
        odom.pose.pose.position.z = translation[2];
        Eigen::Quaterniond q(rotation);
        odom.pose.pose.orientation.w = q.w();
        odom.pose.pose.orientation.x = q.x();
        odom.pose.pose.orientation.y = q.y();
        odom.pose.pose.orientation.z = q.z();
        pubOdomAftMapped.publish(odom);
        loop.sleep();
        cloudInd++;
    }
    double mean_descriptor_time =
            std::accumulate(descriptor_time.begin(), descriptor_time.end(), 0) *
            1.0 / descriptor_time.size();
    double mean_query_time =
            std::accumulate(querying_time.begin(), querying_time.end(), 0) *
            1.0 / querying_time.size();
    double mean_update_time =
            std::accumulate(update_time.begin(), update_time.end(), 0) * 1.0 /
            update_time.size();
    double mean_translation_error =
            std::accumulate(t_error_vec.begin(), t_error_vec.end(), 0) * 1.0 /
            t_error_vec.size();
    double mean_rotation_error =
            std::accumulate(r_error_vec.begin(), r_error_vec.end(), 0) * 1.0 /
            t_error_vec.size();
    std::cout << "Total key frame number:" << keyCloudInd
              << ", loop number:" << triggle_loop_num << std::endl;
    std::cout << "Mean time for descriptor extraction: " << mean_descriptor_time
              << "ms, query: " << mean_query_time
              << "ms, update: " << mean_update_time << "ms, total: "
              << mean_descriptor_time + mean_query_time + mean_update_time
              << "ms" << std::endl;
    std::cout << "Mean translation error: " << mean_translation_error
              << std::endl;
    std::cout << "Mean ratation error   : " << mean_rotation_error << std::endl;
    return 0;
}