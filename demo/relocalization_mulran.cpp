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

pcl::PointCloud<pcl::PointXYZI>::ConstPtr getCloud(std::string filename) {
    FILE *file = fopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "error: failed to load point cloud " << filename
                  << std::endl;
        return nullptr;
    }

    std::vector<float> buffer(1000000);
    size_t num_points = fread(reinterpret_cast<char *>(buffer.data()),
                              sizeof(float), buffer.size(), file) /
                        4;
    fclose(file);

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZI>());
    cloud->resize(num_points);

    for (int i = 0; i < num_points; i++) {
        auto &pt = cloud->at(i);
        pt.x = buffer[i * 4];
        pt.y = buffer[i * 4 + 1];
        pt.z = buffer[i * 4 + 2];
        // Intensity is not in use
        pt.intensity = buffer[i * 4 + 3];
    }

    return cloud;
}

void color_point_cloud(pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_in,
                       std::vector<int> color,
                       pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_out) {
    int r, g, b;
    r = color[0];
    g = color[1];
    b = color[2];
    pcl::PointXYZRGB temp_pt;
    for (int i = 0; i < cloud_in->points.size(); ++i) {
        temp_pt.x = cloud_in->points[i].x;
        temp_pt.y = cloud_in->points[i].y;
        temp_pt.z = cloud_in->points[i].z;
        temp_pt.r = r;
        temp_pt.g = g;
        temp_pt.b = b;
        cloud_out->points.push_back(temp_pt);
    }
}

bool save_result(std::ofstream &out,
                 std::vector<double> result_vec,
                 int index) {
    if (!out) {
        return false;
    }
    std::string space_delimiter = " ";

    out << index << space_delimiter;

    for (size_t i = 0; i < result_vec.size(); ++i) {
        if (i == result_vec.size() - 1) {
            out << result_vec[i];
        } else {
            out << result_vec[i] << space_delimiter;
        }
    }
    out << std::endl;

    return true;
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "demo_mulran");
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
    std::string result_file_path = "";

    nh.param<std::string>("generate_lidar_path", generate_lidar_path, "");
    nh.param<std::string>("generate_pose_path", generate_pose_path, "");
    nh.param<std::string>("localization_lidar_path", localization_lidar_path,
                          "");
    nh.param<std::string>("localization_pose_path", localization_pose_path, "");

    nh.param<bool>("generate_descriptor_database", generateDataBase, false);
    nh.param<std::string>("descriptor_file_path", descriptor_path, "");

    nh.param<std::string>("reference_gt_path", reference_gt_path, "");
    nh.param<std::string>("global_map_path", global_map_path, "");

    nh.param<std::string>("result_file_path", result_file_path, "");
    std::ofstream result_stream(result_file_path);

    std::vector<std::string> genScanFiles;
    // batch_read_filenames_in_folder(generate_lidar_path, "_filelist.txt", ".bin",
    //                                genScanFiles);
    genScanFiles = getFilesInDirectory(generate_lidar_path);
    std::vector<std::string> locScanFiles;
    // batch_read_filenames_in_folder(localization_lidar_path, "_filelist.txt",
    //                                ".bin", locScanFiles);
    locScanFiles = getFilesInDirectory(localization_lidar_path);
    

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

    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
            GT_list_camera;
    GT_list_camera = load_poses_from_transform_matrix(generate_pose_path);

    std::cout << "Pose size is: " << GT_list_camera.size() << std::endl;

    Eigen::Matrix4d calib_mat;

    // NTU MCD
    // clang-format off
    calib_mat << 0.9999346552051229, 0.003477624535771754, -0.010889970036688295, -0.060649229060416594,
                 0.003587143302461965, -0.9999430279821171, 0.010053516443599904, -0.012837544242408117,
                 -0.010854387257665576, -0.01009192338171122, -0.999890161647627, -0.020492606896077407,
                 0.0, 0.0, 0.0, 1.0;
    // clang-format on

    // Mulran
    // clang-format off
//     calib_mat << -0.99998295, 0.00583984, -0.00000524, 1.70430303,
//                  -0.00583984,-0.99998295, 0.00000175, -0.01105054,
//                  -0.00000523, 0.00000178, 1.0,        -1.80469106,
//                   0.0, 0.0, 0.0, 1.0;
    // clang-format on

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

    // size_t gen_total_size = generate_poses_vec.size();
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
        for (int cloudInd = 0; cloudInd < GT_list_camera.size(); ++cloudInd) {
            std::string curPcPath = genScanFiles[cloudInd];
            //     std::cout << "Lidar path:  " << curPcPath << std::endl;

            pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(
                    new pcl::PointCloud<pcl::PointXYZI>());
            *current_cloud = *getCloud(curPcPath);

            Eigen::Matrix4d gt_lidar = GT_list_camera[cloudInd] * calib_mat;
            Eigen::Vector3d translation = gt_lidar.topRightCorner(3, 1);
            Eigen::Matrix3d rotation = gt_lidar.topLeftCorner(3, 3);

            pcl::transformPointCloud(*current_cloud, *current_cloud, gt_lidar);

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
    std_manager->loadExistingSTD(descriptor_path, GT_list_camera.size());
    ROS_INFO("Loaded saved STD.");

    /////////////// localization //////////////
    bool flagStop = false;

    cloudInd = 0;
    std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>>
            ref_gt_list_camera;
    ref_gt_list_camera =
            load_poses_from_transform_matrix(localization_pose_path);

    size_t loc_total_size = ref_gt_list_camera.size();

    ///////////// start retrieval /////////////////
    int success_count = 0;
    for (; cloudInd < loc_total_size; cloudInd += 1) {
        if (cloudInd >= loc_total_size) {
            break;
        }
        keyCloudInd = cloudInd;

        std::string curPcPath = locScanFiles[int(cloudInd)];
        pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud(
                new pcl::PointCloud<pcl::PointXYZI>());
        *current_cloud = *getCloud(curPcPath);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_registered(
                new pcl::PointCloud<pcl::PointXYZI>());

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_registered_colored(
                new pcl::PointCloud<pcl::PointXYZRGB>());
        Eigen::Matrix4d gt_lidar = ref_gt_list_camera[cloudInd] * calib_mat;
        Eigen::Vector3d translation = gt_lidar.topRightCorner(3, 1);
        Eigen::Matrix3d rotation = gt_lidar.topLeftCorner(3, 3);

        down_sampling_voxel(*current_cloud, config_setting.ds_size_);
        for (auto pv : current_cloud->points) {
            temp_cloud->points.push_back(pv);
        }

        // check if keyframe
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
            //     std_manager->PlaneGeomrtricIcp(
            //             std_manager->current_plane_cloud_,
            //             std_manager->plane_cloud_vec_[match_frame],
            //             loop_transform);

            Eigen::Matrix4d estimated_transform;
            estimated_transform.topLeftCorner(3, 3) = loop_transform.second;
            estimated_transform.topRightCorner(3, 1) = loop_transform.first;

            // pcl::transformPointCloud<pcl::PointXYZI>(
            //         *temp_cloud, *cloud_registered,
            //         (estimated_transform).cast<float>());

            Eigen::Matrix4d odom_trans_matrix;
            odom_trans_matrix.setIdentity();
            odom_trans_matrix.topLeftCorner(3, 3) = rotation;
            odom_trans_matrix.topRightCorner(3, 1) = translation;

            Eigen::Matrix4d estimated_pose_inW;
            estimated_pose_inW = estimated_transform;

            Eigen::Vector3d gt_translation = translation;
            Eigen::Matrix3d gt_rotation = rotation;

            double t_e =
                    (gt_translation - estimated_pose_inW.topRightCorner(3, 1))
                            .norm();
            double r_e = std::abs(std::acos(fmin(
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

            bool suc_reg_flag = false;
            if (t_e < 8.5 && r_e < 10.0) {
                success_count++;
                suc_reg_flag = true;
            }
            t_error_vec.push_back(t_e);
            r_error_vec.push_back(r_e);

            Eigen::Matrix4d viz_transform;
            std::vector<int> pc_color;
            if (suc_reg_flag) {
                viz_transform = gt_lidar;
                pc_color = {0, 255, 0};
            } else {
                viz_transform = odom_trans_matrix;
                pc_color = {255, 0, 0};
            }

            pcl::transformPointCloud(*current_cloud, *cloud_registered,
                                     viz_transform);
            color_point_cloud(cloud_registered, pc_color,
                              cloud_registered_colored);

            // if (r_e > 100.0) {
            //     flagStop = true;
            // }
        } else {
            t_error_vec.push_back(200);
            r_error_vec.push_back(100);

            pcl::transformPointCloud(*current_cloud, *cloud_registered,
                                     gt_lidar);
            color_point_cloud(cloud_registered, {255, 0, 0},
                              cloud_registered_colored);
        }
        auto t_query_end = std::chrono::high_resolution_clock::now();
        querying_time.push_back(time_inc(t_query_end, t_query_begin));

        // logging
        std::vector<double> result;
        result.emplace_back(t_error_vec.back());
        result.emplace_back(r_error_vec.back());
        result.emplace_back(querying_time.back());

        if (!save_result(result_stream, result, cloudInd)) {
            ROS_ERROR("Save result file fail!");
            return 0;
        }

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

        //     triggle_loop_num++;
        //     pcl::toROSMsg(*std_manager->plane_cloud_vec_[search_result.first],
        //                   pub_cloud);
        //     pub_cloud.header.frame_id = "camera_init";
        //     pubMatchedCloud.publish(pub_cloud);
        slow_loop.sleep();

        pcl::toROSMsg(*cloud_registered_colored, pub_cloud);
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

        temp_cloud->clear();
        // keyCloudInd += 5;
        // slow_loop.sleep();

        // nav_msgs::Odometry odom;
        // odom.header.frame_id = "camera_init";
        // odom.pose.pose.position.x = translation[0];
        // odom.pose.pose.position.y = translation[1];
        // odom.pose.pose.position.z = translation[2];
        // Eigen::Quaterniond q(rotation);
        // odom.pose.pose.orientation.w = q.w();
        // odom.pose.pose.orientation.x = q.x();
        // odom.pose.pose.orientation.y = q.y();
        // odom.pose.pose.orientation.z = q.z();
        // pubOdomAftMapped.publish(odom);
        loop.sleep();
        // cloudInd += 5;
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
    std::cout << "Total key frame number: " << keyCloudInd
              << ", relocalized number: " << success_count
              << " w/ succuss ratio "
              << (double)success_count / keyCloudInd * 100 << "%." << std::endl;
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