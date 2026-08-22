#include "pose_estimation.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <algorithm>
#include <apriltag/apriltag_pose.h>
#include <apriltag/common/homography.h>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <tf2/convert.hpp>

namespace
{
    constexpr double kDerivativeStep = 1e-6;

    double wrappedDifference(const double a, const double b)
    {
        return std::atan2(std::sin(a - b), std::cos(a - b));
    }

    Eigen::Vector3d rpyFromRvec(const cv::Mat& rvec)
    {
        cv::Mat rotation;
        cv::Rodrigues(rvec, rotation);

        const double roll = std::atan2(rotation.at<double>(2, 1), rotation.at<double>(2, 2));
        const double pitch = std::atan2(
            -rotation.at<double>(2, 0),
            std::hypot(rotation.at<double>(2, 1), rotation.at<double>(2, 2)));
        const double yaw = std::atan2(rotation.at<double>(1, 0), rotation.at<double>(0, 0));
        return {roll, pitch, yaw};
    }

    PoseEstimate pnpFromPointsWithCovariance(const std::vector<cv::Point3d>& object_points,
                                             const std::vector<cv::Point2d>& image_points,
                                             const std::vector<int>& point_tag_ids,
                                             const std::array<double, 4>& intr,
                                             const double pixel_stddev,
                                             const double covariance_scale,
                                             const double max_condition_number)
    {
        PoseEstimate estimate;
        cv::Matx33d camera_matrix = cv::Matx33d::eye();
        camera_matrix(0, 0) = intr[0];
        camera_matrix(1, 1) = intr[1];
        camera_matrix(0, 2) = intr[2];
        camera_matrix(1, 2) = intr[3];

        cv::Mat rvec;
        cv::Mat tvec;
        if(!cv::solvePnP(object_points, image_points, camera_matrix, {}, rvec, tvec)) {
            return estimate;
        }

        estimate.transform = tf2::toMsg<std::pair<cv::Mat_<double>, cv::Mat_<double>>, geometry_msgs::msg::Transform>(std::make_pair(tvec, rvec));
        estimate.valid = true;

        std::vector<cv::Point2d> projected_points;
        cv::Mat projection_jacobian;
        cv::projectPoints(object_points, rvec, tvec, camera_matrix, {}, projected_points, projection_jacobian);

        const Eigen::Index residual_count = static_cast<Eigen::Index>(2 * image_points.size());
        Eigen::MatrixXd jacobian(residual_count, 6);
        Eigen::VectorXd residual(residual_count);
        for(size_t point = 0; point < image_points.size(); ++point) {
            residual(2 * point) = image_points[point].x - projected_points[point].x;
            residual(2 * point + 1) = image_points[point].y - projected_points[point].y;
            for(int parameter = 0; parameter < 6; ++parameter) {
                jacobian(2 * point, parameter) = projection_jacobian.at<double>(2 * point, parameter);
                jacobian(2 * point + 1, parameter) = projection_jacobian.at<double>(2 * point + 1, parameter);
            }
        }

        double squared_reprojection_error = 0.0;
        std::unordered_map<int, double> tag_squared_errors;
        std::unordered_map<int, size_t> tag_point_counts;
        for(size_t point = 0; point < image_points.size(); ++point) {
            const double dx = residual(2 * point);
            const double dy = residual(2 * point + 1);
            const double squared_error = dx * dx + dy * dy;
            squared_reprojection_error += squared_error;
            tag_squared_errors[point_tag_ids[point]] += squared_error;
            ++tag_point_counts[point_tag_ids[point]];
        }
        estimate.reprojection_error = std::sqrt(
            squared_reprojection_error / static_cast<double>(image_points.size()));
        for(const auto& tag_error : tag_squared_errors) {
            const double tag_reprojection_error = std::sqrt(
                tag_error.second / static_cast<double>(tag_point_counts.at(tag_error.first)));
            if(tag_reprojection_error > estimate.worst_tag_reprojection_error) {
                estimate.worst_tag_reprojection_error = tag_reprojection_error;
                estimate.worst_tag_id = tag_error.first;
            }
        }

        const Eigen::Matrix<double, 6, 6> information = jacobian.transpose() * jacobian;
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(information);
        if(eigensolver.info() != Eigen::Success || eigensolver.eigenvalues().minCoeff() <= 0.0) {
            return estimate;
        }
        const double condition_number = eigensolver.eigenvalues().maxCoeff() / eigensolver.eigenvalues().minCoeff();
        if(!std::isfinite(condition_number) || condition_number > max_condition_number) {
            return estimate;
        }

        const double degrees_of_freedom = std::max(1.0, static_cast<double>(residual_count - 6));
        const double fitted_pixel_variance = residual.squaredNorm() / degrees_of_freedom;
        const double pixel_variance = covariance_scale * std::max(pixel_stddev * pixel_stddev, fitted_pixel_variance);
        const Eigen::Matrix<double, 6, 6> pnp_covariance =
            pixel_variance * information.ldlt().solve(Eigen::Matrix<double, 6, 6>::Identity());

        // OpenCV parameterizes pose as [rotation vector, translation]. ROS pose
        // covariance is ordered [x, y, z, fixed-axis roll, pitch, yaw].
        Eigen::Matrix<double, 6, 6> coordinate_jacobian = Eigen::Matrix<double, 6, 6>::Zero();
        coordinate_jacobian.block<3, 3>(0, 3).setIdentity();
        const Eigen::Vector3d nominal_rpy = rpyFromRvec(rvec);
        for(int axis = 0; axis < 3; ++axis) {
            cv::Mat perturbed_rvec = rvec.clone();
            perturbed_rvec.at<double>(axis) += kDerivativeStep;
            const Eigen::Vector3d perturbed_rpy = rpyFromRvec(perturbed_rvec);
            for(int output_axis = 0; output_axis < 3; ++output_axis) {
                coordinate_jacobian(3 + output_axis, axis) =
                    wrappedDifference(perturbed_rpy(output_axis), nominal_rpy(output_axis)) / kDerivativeStep;
            }
        }

        Eigen::Matrix<double, 6, 6> ros_covariance =
            coordinate_jacobian * pnp_covariance * coordinate_jacobian.transpose();
        ros_covariance = 0.5 * (ros_covariance + ros_covariance.transpose());
        if(!ros_covariance.allFinite()) {
            return estimate;
        }
        for(int row = 0; row < 6; ++row) {
            for(int column = 0; column < 6; ++column) {
                estimate.covariance[6 * row + column] = ros_covariance(row, column);
            }
        }
        estimate.covariance_valid = true;
        return estimate;
    }
}// namespace


geometry_msgs::msg::Transform
homography(apriltag_detection_t* const detection, const std::array<double, 4>& intr, double tagsize)
{
    apriltag_detection_info_t info = {detection, tagsize, intr[0], intr[1], intr[2], intr[3]};

    apriltag_pose_t pose;
    estimate_pose_for_tag_homography(&info, &pose);

    // rotate frame such that z points in the opposite direction towards the camera
    for(int i = 0; i < 3; i++) {
        // swap x and y axes
        std::swap(MATD_EL(pose.R, 0, i), MATD_EL(pose.R, 1, i));
        // invert z axis
        MATD_EL(pose.R, 2, i) *= -1;
    }

    return tf2::toMsg<apriltag_pose_t, geometry_msgs::msg::Transform>(const_cast<const apriltag_pose_t&>(pose));
}

geometry_msgs::msg::Transform pnp_from_points(
    const std::vector<cv::Point3d>& objectPoints,
    const std::vector<cv::Point2d>& imagePoints,
    const std::array<double, 4>& intr)
{
    cv::Matx33d cameraMatrix = cv::Matx33d::eye();
    cameraMatrix(0, 0) = intr[0];// fx
    cameraMatrix(1, 1) = intr[1];// fy
    cameraMatrix(0, 2) = intr[2];// cx
    cameraMatrix(1, 2) = intr[3];// cy

    cv::Mat rvec, tvec;
    cv::solvePnP(objectPoints, imagePoints, cameraMatrix, {}, rvec, tvec);

    return tf2::toMsg<std::pair<cv::Mat_<double>, cv::Mat_<double>>, geometry_msgs::msg::Transform>(std::make_pair(tvec, rvec));
}


geometry_msgs::msg::Transform
pnp(apriltag_detection_t* const detection, const std::array<double, 4>& intr, double tagsize)
{
    const std::vector<cv::Point3d> objectPoints{
        {-tagsize / 2, -tagsize / 2, 0},
        {+tagsize / 2, -tagsize / 2, 0},
        {+tagsize / 2, +tagsize / 2, 0},
        {-tagsize / 2, +tagsize / 2, 0},
    };

    const std::vector<cv::Point2d> imagePoints{
        {detection->p[0][0], detection->p[0][1]},
        {detection->p[1][0], detection->p[1][1]},
        {detection->p[2][0], detection->p[2][1]},
        {detection->p[3][0], detection->p[3][1]},
    };

    return pnp_from_points(objectPoints, imagePoints, intr);
}

PoseEstimate
pnp_bundle(std::vector<apriltag_detection_t*> detections,
           const std::array<double, 4>& intr,
           const std::unordered_map<int, double>& tagsizes,
           const std::unordered_map<int, std::vector<double>>& transforms,
           const double pixel_stddev,
           const double covariance_scale,
           const double max_condition_number,
           const double max_reprojection_error,
           const double max_tag_reprojection_error)
{
    const auto estimate_detections = [&](const std::vector<apriltag_detection_t*>& selected_detections) {
        std::vector<cv::Point3d> objectPoints;
        std::vector<cv::Point2d> imagePoints;
        std::vector<int> pointTagIds;

        for(const apriltag_detection_t* detection : selected_detections) {
            int id = detection->id;
            double s = tagsizes.at(id) / 2;
            const std::vector<double>& tf = transforms.at(detection->id);

            // note: The tag bundles should probably store pre-computed bundle-frame tag points rather than recalculating it every detection
            std::vector<Eigen::Vector3d> corners = {
                Eigen::Vector3d(-s, -s, 0),
                Eigen::Vector3d(+s, -s, 0),
                Eigen::Vector3d(+s, +s, 0),
                Eigen::Vector3d(-s, +s, 0)};

            Eigen::Affine3d transform = Eigen::Translation3d(tf[0], tf[1], tf[2]) * Eigen::Quaterniond(tf[3], tf[4], tf[5], tf[6]);

            for(const Eigen::Vector3d& point : corners) {
                Eigen::Vector3d transformed_point = transform * point;
                objectPoints.push_back(cv::Point3d(transformed_point.x(), transformed_point.y(), transformed_point.z()));
                pointTagIds.push_back(id);
            }
            // Add image points
            for(int i = 0; i < 4; i++) {
                imagePoints.push_back({detection->p[i][0], detection->p[i][1]});
            }
        }

        return pnpFromPointsWithCovariance(
            objectPoints, imagePoints, pointTagIds, intr, pixel_stddev, covariance_scale,
            max_condition_number);
    };

    PoseEstimate estimate = estimate_detections(detections);
    if(!estimate.valid) {
        return estimate;
    }

    // A newly visible, bad, or slightly mis-surveyed tag should not poison an
    // otherwise consistent bundle. Retry at most once without the tag whose
    // corners have the largest RMS reprojection error. Keep at least two tags
    // in the solve so normal single- and two-tag behavior is unchanged.
    if(max_tag_reprojection_error > 0.0 && detections.size() >= 3 &&
       estimate.worst_tag_reprojection_error > max_tag_reprojection_error) {
        std::vector<apriltag_detection_t*> filtered_detections;
        filtered_detections.reserve(detections.size() - 1);
        for(apriltag_detection_t* detection : detections) {
            if(detection->id != estimate.worst_tag_id) {
                filtered_detections.push_back(detection);
            }
        }

        PoseEstimate filtered_estimate = estimate_detections(filtered_detections);
        if(filtered_estimate.valid &&
           filtered_estimate.reprojection_error < estimate.reprojection_error) {
            filtered_estimate.rejected_tag_id = estimate.worst_tag_id;
            estimate = filtered_estimate;
        }
    }

    if(max_reprojection_error > 0.0 &&
       estimate.reprojection_error > max_reprojection_error) {
        estimate.valid = false;
        estimate.covariance_valid = false;
    }

    return estimate;
}
const std::unordered_map<std::string, pose_estimation_f> pose_estimation_methods{
    {"homography", homography},
    {"pnp", pnp},
};
