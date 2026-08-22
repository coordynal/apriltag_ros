#include "pose_estimation.hpp"

#include <apriltag/apriltag.h>
#include <cmath>
#include <gtest/gtest.h>
#include <opencv2/calib3d.hpp>
#include <unordered_map>
#include <vector>

namespace
{
    constexpr double kTagSize = 0.16;

    struct SyntheticBundle
    {
        std::vector<apriltag_detection_t> storage;
        std::vector<apriltag_detection_t*> detections;
        std::unordered_map<int, double> sizes;
        std::unordered_map<int, std::vector<double>> transforms;
    };

    SyntheticBundle makeBundle(const int tag_count)
    {
        SyntheticBundle bundle;
        bundle.storage.resize(tag_count);
        const cv::Mat rvec = (cv::Mat_<double>(3, 1) << M_PI, 0.0, 0.0);
        const cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0.0, 0.0, 2.0);
        const cv::Matx33d camera_matrix{
            900.0, 0.0, 640.0,
            0.0, 900.0, 400.0,
            0.0, 0.0, 1.0};

        for(int index = 0; index < tag_count; ++index) {
            const int id = index + 1;
            const double tag_x = 0.3 * (index - (tag_count - 1) / 2.0);
            bundle.sizes[id] = kTagSize;
            bundle.transforms[id] = {tag_x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};

            const double half_size = kTagSize / 2.0;
            const std::vector<cv::Point3d> object_points{
                {tag_x - half_size, -half_size, 0.0},
                {tag_x + half_size, -half_size, 0.0},
                {tag_x + half_size, +half_size, 0.0},
                {tag_x - half_size, +half_size, 0.0}};
            std::vector<cv::Point2d> image_points;
            cv::projectPoints(object_points, rvec, tvec, camera_matrix, {}, image_points);

            apriltag_detection_t& detection = bundle.storage[index];
            detection.id = id;
            for(size_t corner = 0; corner < image_points.size(); ++corner) {
                detection.p[corner][0] = image_points[corner].x;
                detection.p[corner][1] = image_points[corner].y;
            }
        }

        for(apriltag_detection_t& detection : bundle.storage) {
            bundle.detections.push_back(&detection);
        }
        return bundle;
    }

    PoseEstimate estimate(const SyntheticBundle& bundle,
                          const double max_reprojection_error,
                          const double max_tag_reprojection_error)
    {
        return pnp_bundle(
            bundle.detections, {900.0, 900.0, 640.0, 400.0}, bundle.sizes,
            bundle.transforms, 0.75, 1.0, 1.0e12, max_reprojection_error,
            max_tag_reprojection_error);
    }
}// namespace

TEST(PoseEstimation, CleanTwoTagBundleIsUnchanged)
{
    const SyntheticBundle bundle = makeBundle(2);
    const PoseEstimate result = estimate(bundle, 3.0, 3.0);

    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.rejected_tag_id, -1);
    EXPECT_LT(result.reprojection_error, 1.0e-5);
}

TEST(PoseEstimation, RejectsBadWholeBundleFit)
{
    SyntheticBundle bundle = makeBundle(3);
    for(int corner = 0; corner < 4; ++corner) {
        bundle.storage[2].p[corner][0] += 20.0;
    }

    const PoseEstimate result = estimate(bundle, 3.0, 0.0);
    EXPECT_FALSE(result.valid);
    EXPECT_GT(result.reprojection_error, 3.0);
}

TEST(PoseEstimation, RetriesWithoutWorstTag)
{
    SyntheticBundle bundle = makeBundle(3);
    for(int corner = 0; corner < 4; ++corner) {
        bundle.storage[2].p[corner][0] += 20.0;
    }

    const PoseEstimate result = estimate(bundle, 3.0, 3.0);
    ASSERT_TRUE(result.valid);
    EXPECT_EQ(result.rejected_tag_id, 3);
    EXPECT_LT(result.reprojection_error, 1.0e-5);
}
