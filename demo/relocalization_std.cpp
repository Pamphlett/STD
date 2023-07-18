#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include "include/utility.h"
#include "include/STDesc.h"

using namespace pcl;
using namespace Eigen;
using namespace Util;

// Read KITTI data
std::vector<float> read_lidar_data(const std::string lidar_data_path)
{
    std::ifstream lidar_data_file;
    lidar_data_file.open(lidar_data_path,
                         std::ifstream::in | std::ifstream::binary);
    if (!lidar_data_file)
    {
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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "demo_ntu");
    ros::NodeHandle nh;


    /* #region For online validation --------------------------------------------------------------------------------*/

    std::string reference_gt_path = "";
    nh.param<std::string>("reference_gt_path", reference_gt_path, "");

    /* #endregion For online validation -----------------------------------------------------------------------------*/


    /* #region Declaration to database on prior map -----------------------------------------------------------------*/

    // Path to load / save descriptor
    std::string descriptor_path = "";
    nh.param<std::string>("descriptor_file_path", descriptor_path, "");

    bool generateDataBase;
    nh.param<bool>("generate_descriptor_database", generateDataBase, false);

    std::string generated_lidar_path = "";
    nh.param<std::string>("generated_lidar_path", generated_lidar_path, "");

    std::string generated_pose_path = "";
    nh.param<std::string>("generated_pose_path", generated_pose_path, "");

    string database_type = (generated_pose_path.find(string(".csv")) != std::string::npos) ? "csv" : "pcd";

    /* #endregion Declaration to database on prior map --------------------------------------------------------------*/


    /* #region Recorded data of online localization -----------------------------------------------------------------*/

    std::string localization_lidar_path = "";
    nh.param<std::string>("localization_lidar_path", localization_lidar_path, "");

    std::string localization_pose_path = "";
    nh.param<std::string>("localization_pose_path", localization_pose_path, "");

    /* #endregion Recorded data of online localization --------------------------------------------------------------*/


    // Read the config
    ConfigSetting config_setting;
    read_parameters(nh, config_setting);


    /* #region Create the publishers --------------------------------------------------------------------------------*/

    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
    ros::Publisher pubRegisterCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
    ros::Publisher pubCurrentCloud  = nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
    ros::Publisher pubCurrentCorner = nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
    ros::Publisher pubMatchedCloud  = nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
    ros::Publisher pubMatchedCorner = nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
    ros::Publisher pubSTD           = nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

    /* #endregion Create the publishers -----------------------------------------------------------------------------*/


    // Create the detector
    STDescManager *std_manager = new STDescManager(config_setting);


    /* #region Generate the data base -------------------------------------------------------------------------------*/

    if (generateDataBase)
    {
        /* #region Load the database ------------------------------------------------------------------------------------*/

        std::vector<int> generated_index_vec;
        std::vector<std::pair<Vector3d, Matrix3d>> generated_poses_vec;
        std::vector<std::string> generated_times_vec;
        std::vector<std::pair<Vector3d, Matrix3d>>generated_key_poses_vec;

        if (database_type == "csv")
        {
            load_CSV_pose_with_time(generated_pose_path, generated_index_vec,
                                    generated_poses_vec, generated_times_vec);
        }
        else
        {
            std::vector<double> generated_times_vec_;
            load_keyframes_pose_pcd(generated_pose_path, generated_index_vec,
                                    generated_poses_vec, generated_times_vec_);
            for(int idx : generated_index_vec)
                generated_times_vec.push_back(std::to_string(idx));
        }


        std::cout << "Sucessfully load pose with number: "
                  << generated_poses_vec.size() << std::endl;

        /* #endregion load the database ---------------------------------------------------------------------------------*/

        size_t gen_total_size = generated_poses_vec.size();
        CloudXYZIPtr temp_cloud(new CloudXYZI());

        // generate descriptor database
        printf("Generating descriptors ...");

        if(database_type == "csv")
        {
            for (int cloudInd = 0; cloudInd < gen_total_size; ++cloudInd)
            {
                std::string ori_time_str = generated_times_vec[cloudInd];
                std::replace(ori_time_str.begin(), ori_time_str.end(), '.', '_');
                std::string curr_lidar_path = generated_lidar_path + "cloud_" + std::to_string(cloudInd + 1) + "_" + ori_time_str + ".pcd";

                CloudXYZIPtr current_cloud(new CloudXYZI());

                if (pcl::io::loadPCDFile<PointXYZI>(curr_lidar_path, *current_cloud) == -1)
                {
                    ROS_ERROR("Couldn't read scan from file. \n");
                    std::cout << "Current File Name is: "
                              << generated_index_vec[cloudInd] << std::endl;
                    return (-1);
                }

                myTf tf_W_B(generated_poses_vec[cloudInd].second, generated_poses_vec[cloudInd].first);
                pcl::transformPointCloud<PointXYZI>(*current_cloud, *current_cloud, tf_W_B.cast<float>().tfMat());
                down_sampling_voxel(*current_cloud, config_setting.ds_size_);

                if (cloudInd % config_setting.sub_frame_num_ == 0)
                {
                    // step1. Descriptor Extraction
                    // auto t_descriptor_begin =
                    // std::chrono::high_resolution_clock::now();
                    std::vector<STDesc> stds_vec;
                    std_manager->GenerateSTDescs(current_cloud, stds_vec);

                    // step3. Add descriptors to the database
                    // auto t_map_update_begin =
                    // std::chrono::high_resolution_clock::now();
                    std_manager->AddSTDescs(stds_vec);
                    // auto t_map_update_end =
                    // std::chrono::high_resolution_clock::now();
                    // update_time.push_back(
                    //     time_inc(t_map_update_end, t_map_update_begin));

                    // CloudXYZI save_key_cloud;
                    // save_key_cloud = *temp_cloud;

                    // std_manager->key_cloud_vec_.push_back(
                    //     save_key_cloud.makeShared());
                    // generated_key_poses_vec.push_back(generated_poses_vec[cloudInd]);
                }

                if (cloudInd % 100 == 0)
                {
                    ROS_INFO("Generated %d frames", cloudInd);
                }
                // temp_cloud->clear();
            }
        }
        else
        {
            for (int cloudInd = 0; cloudInd < gen_total_size; ++cloudInd)
            {
                std::string ori_time_str = generated_times_vec[cloudInd];
                std::replace(ori_time_str.begin(), ori_time_str.end(), '.', '_');
                std::string curr_lidar_path = generated_lidar_path + "KfFullPcl_" + std::to_string(cloudInd) + ".pcd";

                printf("Reading scan: %s\n", curr_lidar_path.c_str());

                CloudXYZIPtr current_cloud(new CloudXYZI());

                if (pcl::io::loadPCDFile<PointXYZI>(curr_lidar_path, *current_cloud) == -1)
                {
                    ROS_ERROR("Couldn't read scan from file. \n");
                    std::cout << "Current File Name is: "
                              << generated_index_vec[cloudInd] << std::endl;
                    return (-1);
                }

                // myTf tf_W_B(generated_poses_vec[cloudInd].second, generated_poses_vec[cloudInd].first);
                // pcl::transformPointCloud<PointXYZI>(*current_cloud, *current_cloud, tf_W_B.cast<float>().tfMat());
                down_sampling_voxel(*current_cloud, config_setting.ds_size_);

                if (cloudInd % config_setting.sub_frame_num_ == 0)
                {
                    // step1. Descriptor Extraction
                    std::vector<STDesc> stds_vec;
                    std_manager->GenerateSTDescs(current_cloud, stds_vec);

                    // step3. Add descriptors to the database
                    std_manager->AddSTDescs(stds_vec);
                }

                ROS_INFO("Generated %d frames", cloudInd);
                // temp_cloud->clear();
            }
        }

        // save generated things
        std_manager->saveToFile(descriptor_path);

        ROS_INFO("Generation done. Exiting program ... ");
        return 0;
    }

    // load STD descriptors in storage
    std_manager->loadExistingSTD(descriptor_path);
    ROS_INFO("Loaded saved STD.");

    /* #endregion Generate the data base -------------------------------------------------------------------------------*/


    /////////////// localization //////////////


    bool flagStop = false;
    CloudXYZIPtr cloud_registered(new CloudXYZI());


    /* #region Loading the ground truth for reference ---------------------------------------------------------------*/

    std::vector<int> reference_index_vec;
    std::vector<std::pair<Vector3d, Matrix3d>> reference_gt_pose_vec;
    std::vector<std::string> reference_time_vec_string;
    std::vector<double> reference_time_vec;

    load_CSV_pose_with_time(reference_gt_path, reference_index_vec,
                            reference_gt_pose_vec, reference_time_vec_string);

    for (auto itr : reference_time_vec_string)
        reference_time_vec.push_back(std::stod(itr));

    /* #endregion Loading the ground truth for reference ------------------------------------------------------------*/


    /* #region Load the localization data ---------------------------------------------------------------------------*/

    std::vector<std::pair<Vector3d, Matrix3d>> localization_poses_vec;
    std::vector<double> key_frame_times_vec;
    std::vector<int> localization_index_vec;
    std::vector<std::pair<Vector3d, Matrix3d>> localization_key_poses_vec;
    load_keyframes_pose_pcd(localization_pose_path, localization_index_vec,
                            localization_poses_vec, key_frame_times_vec);

    std::cout << "Sucessfully loaded pointclouds to be relocalized: "
              << localization_poses_vec.size() << std::endl;

    size_t loc_total_size = localization_poses_vec.size();

    std::vector<int> resulted_index = findCorrespondingFrames(key_frame_times_vec, reference_time_vec);

    if (resulted_index.size() != key_frame_times_vec.size())
        ROS_ERROR("KF AND GNDTR SIZE NOT MATCH !");

    /* #endregion Load the localization data ------------------------------------------------------------------------*/
    
    
    ///////////// start retrieval /////////////////
    
    ros::Rate loop(500);
    ros::Rate slow_loop(10);

    std::vector<double> descriptor_time;
    std::vector<double> querying_time;
    std::vector<double> update_time;
    std::vector<double> t_error_vec;
    std::vector<double> r_error_vec;
    int triggle_loop_num = 0;

    int cloudInd = 0;
    size_t keyCloudInd = 0;

    while (ros::ok())
    {
        if (cloudInd >= loc_total_size)
            break;

        /* #region Load the pointcloud from localization process ----------------------------------------------------*/

        // Deduce the correct file name
        int curr_index = localization_index_vec[cloudInd];
        std::string curr_lidar_path = localization_lidar_path + "/KfCloudinW_" + zeroPaddedString(curr_index, loc_total_size) + ".pcd";

        // Load the pointcloud from memory. SHOULD be replaced subscribing to pointcloud odometry 
        CloudXYZIPtr current_cloud(new CloudXYZI());
        CloudXYZIPtr cloud_registered(new CloudXYZI());
        if (pcl::io::loadPCDFile<PointXYZI>(curr_lidar_path, *current_cloud) == -1)
        {
            ROS_ERROR(KRED "Couldn't read scan from file. \n" RESET);
            std::cout << "Current File Name is: "
                      << localization_index_vec[cloudInd] << std::endl;
            return (-1);
        }

        // Extract the pointcloud pose
        myTf tf_W_B(localization_poses_vec[cloudInd].second, localization_poses_vec[cloudInd].first);

        // Transform pointcloud to world frame
        pcl::transformPointCloud<PointXYZI>( *current_cloud, *current_cloud, tf_W_B.inverse().tfMat().cast<float>());

        // Downsample the pointcloud
        down_sampling_voxel(*current_cloud, config_setting.ds_size_);

        /* #endregion Load the pointcloud from localization process -------------------------------------------------*/


        // check if keyframe
        if (cloudInd % config_setting.sub_frame_num_ == 0)
        {
            std::cout << "Key Frame id: " << keyCloudInd
                      << ", cloud size: " << current_cloud->size() << std::endl;
            
            // step1. Descriptor Extraction

            TicToc tt_desc_extr;
            
            std::vector<STDesc> stds_vec;
            std_manager->GenerateSTDescsOneTime(current_cloud, stds_vec);

            descriptor_time.push_back(tt_desc_extr.Toc());
            
            // step2. Searching Loop
            
            TicToc tt_query;

            std::pair<int, double> search_result(-1, 0);
            std::pair<Vector3d, Matrix3d> loop_transform = make_pair(Vector3d(0, 0, 0), Matrix3d::Identity());
            std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
            std_manager->SearchLoop(stds_vec, search_result, loop_transform, loop_std_pair);

            if (search_result.first > 0)
            {
                printf(KGRN "[Loop Detection] triggle loop: %4d -- %4d. Score: %.5f\n" RESET, keyCloudInd, search_result.first, search_result.second);

                // Compute Pose Estimation Error
                int match_frame = search_result.first;
                std_manager->PlaneGeomrtricIcp(std_manager->current_plane_cloud_,
                                               std_manager->plane_cloud_vec_[match_frame], loop_transform);

                myTf tf_W_B_est(loop_transform.second, loop_transform.first);

                pcl::transformPointCloud<PointXYZI>(*current_cloud, *cloud_registered, tf_W_B_est.cast<float>().tfMat());

                Vector3d gt_translation = reference_gt_pose_vec[resulted_index[keyCloudInd]].first;
                Matrix3d gt_rotation = reference_gt_pose_vec[resulted_index[keyCloudInd]].second;

                double t_e = (gt_translation - tf_W_B_est.pos).norm();
                double r_e = fabs(wrapTo360(SO3Log(gt_rotation * tf_W_B_est.rot.inverse()).norm() * 180 / M_PI + 180.0) - 180.0);

                printf(KGRN "Estimated Trans Err:  %6.3f m. Rot Err: %6.3f deg.\n" RESET, t_e, r_e);
                
                t_error_vec.push_back(t_e);
                r_error_vec.push_back(r_e);
                
                if (r_e > 100.0)
                    flagStop = true;
            }

            querying_time.push_back(tt_query.Toc());

            // step3. Add descriptors to the database
            std::cout << "[Time] descriptor extraction: "
                      << tt_query.GetLastStop()
                      << "ms, "
                      << "query: " << tt_query.GetLastStop()
                      << "ms, "
                      << "update map:"
                      << " Nan "
                      << "ms" << std::endl;
            std::cout << std::endl;

            // publish
            publishCloud(pubCurrentCloud, *current_cloud, ros::Time::now(), "camera_init");
            publishCloud(pubCurrentCorner, *std_manager->corner_cloud_vec_.back(), ros::Time::now(), "camera_init");

            if (search_result.first >= 0)
            {
                triggle_loop_num++;
                publishCloud(pubMatchedCloud, *std_manager->plane_cloud_vec_[search_result.first], ros::Time::now(), "camera_init");
                slow_loop.sleep();
                
                publishCloud(pubRegisterCloud, *cloud_registered, ros::Time::now(), "camera_init");
                slow_loop.sleep();

                // pcl::toROSMsg(
                //     *std_manager->corner_cloud_vec_[search_result.first],
                //     pub_cloud);
                // pub_cloud.header.frame_id = "camera_init";
                // pubMatchedCorner.publish(pub_cloud);
                // publish_std_pairs(loop_std_pair, pubSTD);
                // slow_loop.sleep();
                // if (flagStop)
                // {
                    getchar();
                    flagStop = false;
                // }
            }

            // temp_cloud->clear();
            keyCloudInd++;
            slow_loop.sleep();
        }
        
        // Create odom for visualization
        nav_msgs::Odometry odom;
        odom.header.frame_id = "camera_init";
        odom.pose.pose.position.x    = tf_W_B.pos(0);
        odom.pose.pose.position.y    = tf_W_B.pos(1);
        odom.pose.pose.position.z    = tf_W_B.pos(2);
        odom.pose.pose.orientation.w = tf_W_B.rot.w();
        odom.pose.pose.orientation.x = tf_W_B.rot.x();
        odom.pose.pose.orientation.y = tf_W_B.rot.y();
        odom.pose.pose.orientation.z = tf_W_B.rot.z();
        pubOdomAftMapped.publish(odom);
        
        // Increment
        cloudInd++;

        loop.sleep();
    }

    double mean_descriptor_time   = std::accumulate(descriptor_time.begin(), descriptor_time.end(), 0) * 1.0 / descriptor_time.size();
    double mean_query_time        = std::accumulate(querying_time.begin(), querying_time.end(), 0) * 1.0 / querying_time.size();
    double mean_update_time       = std::accumulate(update_time.begin(), update_time.end(), 0) * 1.0 / update_time.size();
    double mean_translation_error = std::accumulate(t_error_vec.begin(), t_error_vec.end(), 0) * 1.0 / t_error_vec.size();
    double mean_rotation_error    = std::accumulate(r_error_vec.begin(), r_error_vec.end(), 0) * 1.0 / t_error_vec.size();

    std::cout << "Total key frame number:"   << keyCloudInd
              << ", loop number:"            << triggle_loop_num << std::endl;
    std::cout << "Mean time for desc extr: " << mean_descriptor_time
              << "ms, query: "               << mean_query_time
              << "ms, update: "              << mean_update_time
              << "ms, total: "               << mean_descriptor_time + mean_query_time + mean_update_time
              << "ms"                        << std::endl;
    std::cout << "Mean translation error: "  << mean_translation_error << std::endl;
    std::cout << "Mean ratation error   : "  << mean_rotation_error << std::endl;

    return 0;
}