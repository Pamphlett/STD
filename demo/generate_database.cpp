#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>

#include "include/utility.h"
#include "include/STDesc.h"

using namespace pcl;
using namespace Eigen;
using namespace Util;

void load_kf_pose_from_pcd(
    const string &pose_file, vector<int> &index_vec, vector<pair<Vector3d, Matrix3d>> &poses_vec, vector<double> &times_vec)
{
    CloudPosePtr CloudPosePtr(new pcl::PointCloud<PointPose>());
    pcl::io::loadPCDFile<PointPose>(pose_file, *CloudPosePtr);

    for (int i = 0; i < CloudPosePtr->points.size(); ++i)
    {
        PointPose p = CloudPosePtr->points[i];
        poses_vec.push_back(make_pair(Vector3d(p.x, p.y, p.z), Quaterniond(p.qw, p.qx, p.qy, p.qz).toRotationMatrix()));
        index_vec.push_back(p.intensity);
        times_vec.push_back(p.t);
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "generate_database");
    ros::NodeHandle nh;

    // Read the config
    ConfigSetting config_setting;
    read_parameters(nh, config_setting);

    // Create the detector
    STDescManager *std_manager = new STDescManager(config_setting);

    /* #region Declaration to database on prior map -----------------------------------------------------------------*/

    bool generateDataBase;
    nh.param<bool>("generate_descriptor_database", generateDataBase, false);

    // Path to load / save descriptor
    std::string descriptor_path = "";
    nh.param<std::string>("descriptor_file_path", descriptor_path, "");

    std::string generated_lidar_path = "";
    nh.param<std::string>("generated_lidar_path", generated_lidar_path, "");

    std::string generated_pose_path = "";
    nh.param<std::string>("generated_pose_path", generated_pose_path, "");

    int kf_merged = 10;
    nh.param<int>("kf_merged", kf_merged, 10);

    cout << "Merging kf: " <<  kf_merged << endl;

    if (generated_pose_path.find(string(".pcd")) == std::string::npos)
        printf("Wrong pose log. Please use a pcd\n");

    /* #endregion Generate the data base ----------------------------------------------------------------------------*/


    /* #region Load the poses ---------------------------------------------------------------------------------------*/

    std::vector<int>                           generated_index_vec;
    std::vector<std::pair<Vector3d, Matrix3d>> generated_poses_vec;
    std::vector<double>                        generated_times_vec;
    std::vector<std::pair<Vector3d, Matrix3d>> generated_key_poses_vec;

    load_kf_pose_from_pcd(generated_pose_path, generated_index_vec,
                          generated_poses_vec, generated_times_vec);

    std::cout << "Sucessfully load pose with number: "
              << generated_poses_vec.size() << std::endl;

    for(int idx = 0; idx < generated_index_vec.size(); idx++)
        printf("Idx: %d. Time: %f\n", idx, generated_times_vec[idx]);

    /* #endregion load the poses --------------------------------------------------------------------------------*/

    size_t gen_total_size = generated_poses_vec.size();

    // generate descriptor database
    printf("Generating descriptors ...");
    {
        for (int cloudInd = 0; cloudInd < gen_total_size; cloudInd += kf_merged)
        {
            CloudXYZIPtr kfCloud(new CloudXYZI());

            for (int k = 0; k < kf_merged && cloudInd + k < gen_total_size; k++)
            {
                std::string ori_time_str = std::to_string(generated_times_vec[cloudInd + k]);
                std::string curr_lidar_path = generated_lidar_path + "/KfCloudinW_" + zeroPaddedString(cloudInd + k, gen_total_size) + ".pcd";
    
                CloudXYZIPtr kfCloud_(new CloudXYZI());

                printf("Reading scan: %s\n", curr_lidar_path.c_str());

                if (pcl::io::loadPCDFile<PointXYZI>(curr_lidar_path, *kfCloud_) == -1)
                {
                    ROS_ERROR("Couldn't read scan from file. \n");
                    std::cout << "Current File Name is: "
                              << generated_index_vec[cloudInd + k] << std::endl;
                    return (-1);
                }

                *kfCloud += *kfCloud_;
            }

            // myTf tf_W_B(generated_poses_vec[cloudInd].second, generated_poses_vec[cloudInd].first);
            // pcl::transformPointCloud<PointXYZI>(*kfCloud, *kfCloud, tf_W_B.inverse().cast<float>().tfMat());
            down_sampling_voxel(*kfCloud, config_setting.ds_size_);

            if (cloudInd % config_setting.sub_frame_num_ == 0)
            {
                // step1. Descriptor Extraction
                std::vector<STDesc> stds_vec;
                std_manager->GenerateSTDescs(kfCloud, stds_vec);

                // step3. Add descriptors to the database
                std_manager->AddSTDescs(stds_vec);
            }

            ROS_INFO("Generated %d frames", cloudInd);
        }
    }

    // save generated things
    std_manager->saveToFile(descriptor_path);

    ROS_INFO("Generation done. Exiting program ... ");
    return 0;

    /* #endregion Generate the data base ----------------------------------------------------------------------------*/

}