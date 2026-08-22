#pragma once

#include <apriltag/apriltag.h>
#include <array>
#include <functional>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <unordered_map>


typedef std::function<geometry_msgs::msg::Transform(apriltag_detection_t* const, const std::array<double, 4>&, const double&)> pose_estimation_f;

struct PoseEstimate
{
    geometry_msgs::msg::Transform transform;
    std::array<double, 36> covariance{};
    double reprojection_error = 0.0;
    double worst_tag_reprojection_error = 0.0;
    int worst_tag_id = -1;
    int rejected_tag_id = -1;
    bool valid = false;
    bool covariance_valid = false;
};

extern const std::unordered_map<std::string, pose_estimation_f> pose_estimation_methods;

PoseEstimate pnp_bundle(std::vector<apriltag_detection_t*> detections,
                        const std::array<double, 4>& intr,
                        const std::unordered_map<int, double>& tagsizes,
                        const std::unordered_map<int, std::vector<double>>& transforms,
                        double pixel_stddev,
                        double covariance_scale,
                        double max_condition_number,
                        double max_reprojection_error,
                        double max_tag_reprojection_error);
