// ros
#include "pose_estimation.hpp"
#include <apriltag_msgs/msg/april_tag_detection.hpp>
#include <apriltag_msgs/msg/april_tag_detection_array.hpp>
#include <cmath>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#ifdef cv_bridge_HPP
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <image_transport/camera_subscriber.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.hpp>

// apriltag
#include "tag_functions.hpp"
#include <apriltag.h>


#define IF(N, V) \
    if(assign_check(parameter, N, V)) continue;

template<typename T>
void assign(const rclcpp::Parameter& parameter, T& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
void assign(const rclcpp::Parameter& parameter, std::atomic<T>& var)
{
    var = parameter.get_value<T>();
}

template<typename T>
bool assign_check(const rclcpp::Parameter& parameter, const std::string& name, T& var)
{
    if(parameter.get_name() == name) {
        assign(parameter, var);
        return true;
    }
    return false;
}

rcl_interfaces::msg::ParameterDescriptor
descr(const std::string& description, const bool& read_only = false)
{
    rcl_interfaces::msg::ParameterDescriptor descr;

    descr.description = description;
    descr.read_only = read_only;

    return descr;
}

const static std::unordered_map<std::string, rmw_qos_profile_t> qos_profiles{
    {"default", rmw_qos_profile_default},
    {"sensor_data", rmw_qos_profile_sensor_data},
    {"system_default", rmw_qos_profile_system_default},
};

struct TagBundle
{
    std::set<int64_t> ids;
    std::unordered_map<int, double> id_to_size;
    std::unordered_map<int, std::vector<double>> id_to_tf;
    std::string frame_id;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pose_publisher;
};

class AprilTagNode : public rclcpp::Node {
public:
    AprilTagNode(const rclcpp::NodeOptions& options);

    ~AprilTagNode() override;

    typedef std::shared_ptr<TagBundle> TagBundlePtr;
    typedef std::vector<TagBundlePtr> TagBundleVec;

private:
    const OnSetParametersCallbackHandle::SharedPtr cb_parameter;

    apriltag_family_t* tf;
    apriltag_detector_t* const td;

    // parameter
    std::mutex mutex;
    double tag_edge_size;
    std::atomic<int> max_hamming;
    std::atomic<bool> profile;
    std::atomic<double> max_tag_distance;
    std::atomic<double> covariance_pixel_stddev;
    std::atomic<double> covariance_scale;
    std::atomic<double> covariance_max_condition_number;
    std::atomic<double> max_reprojection_error;
    std::atomic<double> max_tag_reprojection_error;
    TagBundleVec all_tag_bundles;
    std::unordered_map<int64_t, TagBundleVec> tag_id_to_bundles;
    std::unordered_map<int, std::string> tag_frames;
    std::unordered_map<int, double> tag_sizes;

    std::function<void(apriltag_family_t*)> tf_destructor;

    const image_transport::CameraSubscriber sub_cam;
    const rclcpp::Publisher<apriltag_msgs::msg::AprilTagDetectionArray>::SharedPtr pub_detections;
    tf2_ros::TransformBroadcaster tf_broadcaster;

    pose_estimation_f estimate_pose = nullptr;

    void onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img, const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci);

    rcl_interfaces::msg::SetParametersResult onParameter(const std::vector<rclcpp::Parameter>& parameters);
};

RCLCPP_COMPONENTS_REGISTER_NODE(AprilTagNode)


AprilTagNode::AprilTagNode(const rclcpp::NodeOptions& options)
  : Node("apriltag", options),
    // parameter
    cb_parameter(add_on_set_parameters_callback(std::bind(&AprilTagNode::onParameter, this, std::placeholders::_1))),
    td(apriltag_detector_create()),
    // topics
    sub_cam{
#ifdef image_transport_NODE_INTERFACE
        image_transport::RequiredInterfaces{*this},
#else
        this,
#endif
        this->get_node_topics_interface()->resolve_topic_name("image_rect"),
        std::bind(&AprilTagNode::onCamera, this, std::placeholders::_1, std::placeholders::_2),
        declare_parameter("image_transport", "raw", descr({}, true)),
#ifdef image_transport_QoS
        rclcpp::QoS{rclcpp::QoSInitialization::from_rmw(
#endif
            qos_profiles.at(declare_parameter("qos_profile", "default", descr("qos profile to use. 'default', 'sensor_data' or 'system_default'", true)))
#ifdef image_transport_QoS
                )}
#endif
    },
    pub_detections(create_publisher<apriltag_msgs::msg::AprilTagDetectionArray>("detections", rclcpp::QoS(1))),
    tf_broadcaster(
#ifdef tf2_ros_NODE_INTERFACE
        tf2_ros::TransformBroadcaster::RequiredInterfaces { *this }
#else
        this
#endif
    )
{
    // read-only parameters
    const std::string tag_family = declare_parameter("family", "36h11", descr("tag family", true));
    tag_edge_size = declare_parameter("size", 1.0, descr("default tag size", true));

    // get tag names, IDs and sizes
    const auto ids = declare_parameter("tag.ids", std::vector<int64_t>{}, descr("tag ids", true));
    const auto frames = declare_parameter("tag.frames", std::vector<std::string>{}, descr("tag frame names per id", true));
    const auto sizes = declare_parameter("tag.sizes", std::vector<double>{}, descr("tag sizes per id", true));

    // get list of tag bundles names
    const auto bundle_names = declare_parameter("tag_bundles.bundle_names", std::vector<std::string>{}, descr("tag bundle names", true));
    // get method for estimating tag pose
    const std::string& pose_estimation_method =
        declare_parameter("pose_estimation_method", "pnp",
                          descr("pose estimation method: \"pnp\" (more accurate) or \"homography\" (faster), "
                                "set to \"\" (empty) to disable pose estimation",
                                true));

    if(!pose_estimation_method.empty()) {
        if(pose_estimation_methods.count(pose_estimation_method)) {
            estimate_pose = pose_estimation_methods.at(pose_estimation_method);
        }
        else {
            RCLCPP_ERROR_STREAM(get_logger(), "Unknown pose estimation method '" << pose_estimation_method << "'.");
        }
    }

    // detector parameters in "detector" namespace
    declare_parameter("detector.threads", td->nthreads, descr("number of threads"));
    declare_parameter("detector.decimate", td->quad_decimate, descr("decimate resolution for quad detection"));
    declare_parameter("detector.blur", td->quad_sigma, descr("sigma of Gaussian blur for quad detection"));
    declare_parameter("detector.refine", td->refine_edges, descr("snap to strong gradients"));
    declare_parameter("detector.sharpening", td->decode_sharpening, descr("sharpening of decoded images"));
    declare_parameter("detector.debug", td->debug, descr("write additional debugging images to working directory"));

    declare_parameter("max_hamming", 0, descr("reject detections with more corrected bits than allowed"));
    declare_parameter("profile", false, descr("print profiling information to stdout"));
    max_tag_distance = declare_parameter("max_tag_distance", 3.0,
                                         descr("maximum estimated distance (meters) from the camera for a tag to be included in tag bundle pose estimation; <= 0 disables this filter"));
    covariance_pixel_stddev = declare_parameter(
        "covariance.pixel_stddev", 0.75,
        descr("minimum AprilTag corner standard deviation in pixels used by PnP covariance"));
    covariance_scale = declare_parameter(
        "covariance.scale", 1.0,
        descr("empirical multiplier applied to PnP covariance"));
    covariance_max_condition_number = declare_parameter(
        "covariance.max_condition_number", 1.0e12,
        descr("reject covariance output when the PnP information matrix is more ill-conditioned than this"));
    max_reprojection_error = declare_parameter(
        "max_reprojection_error", 3.0,
        descr("reject bundle poses whose corner RMS reprojection error exceeds this many pixels; <= 0 disables"));
    max_tag_reprojection_error = declare_parameter(
        "max_tag_reprojection_error", 3.0,
        descr("retry bundle PnP once without the worst tag when its corner RMS reprojection error exceeds this many pixels; <= 0 disables"));
    const std::string bundle_pose_topic_prefix = declare_parameter(
        "bundle_pose_topic_prefix", "bundle_poses",
        descr("topic prefix for bundle PoseWithCovarianceStamped measurements", true));

    for(const std::string& bundle_name : bundle_names) {
        TagBundlePtr bundle = std::make_shared<TagBundle>();

        bundle->frame_id = bundle_name;
        bundle->pose_publisher = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            bundle_pose_topic_prefix + "/" + bundle_name, rclcpp::QoS(1));
        std::vector<int64_t> bundle_ids_vector = declare_parameter("tag_bundles." + bundle_name + ".ids", std::vector<int64_t>{}, descr("bundle ids", true));
        bundle->ids = std::set<int64_t>(bundle_ids_vector.begin(), bundle_ids_vector.end());
        for(const int64_t& id : bundle->ids) {
            tag_id_to_bundles[id].push_back(bundle);
            const std::string prefix = "tag_bundles." + bundle_name + "." + std::to_string(id);
            bundle->id_to_size[id] = declare_parameter(prefix + ".size", 1.0, descr("bundle size", true));
            bundle->id_to_tf[id] = declare_parameter(prefix + ".transform", std::vector<double>{}, descr("bundle transform", true));
        }
        all_tag_bundles.push_back(bundle);
    }

    if(!frames.empty()) {
        if(ids.size() != frames.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and frames (" + std::to_string(frames.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_frames[ids[i]] = frames[i]; }
    }

    if(!sizes.empty()) {
        // use tag specific size
        if(ids.size() != sizes.size()) {
            throw std::runtime_error("Number of tag ids (" + std::to_string(ids.size()) + ") and sizes (" + std::to_string(sizes.size()) + ") mismatch!");
        }
        for(size_t i = 0; i < ids.size(); i++) { tag_sizes[ids[i]] = sizes[i]; }
    }

    if(tag_fun.count(tag_family)) {
        tf = tag_fun.at(tag_family).first();
        tf_destructor = tag_fun.at(tag_family).second;
        apriltag_detector_add_family(td, tf);
    }
    else {
        throw std::runtime_error("Unsupported tag family: " + tag_family);
    }
}

AprilTagNode::~AprilTagNode()
{
    apriltag_detector_destroy(td);
    tf_destructor(tf);
}

void AprilTagNode::onCamera(const sensor_msgs::msg::Image::ConstSharedPtr& msg_img,
                            const sensor_msgs::msg::CameraInfo::ConstSharedPtr& msg_ci)
{
    // camera intrinsics for rectified images
    const std::array<double, 4> intrinsics = {msg_ci->p[0], msg_ci->p[5], msg_ci->p[2], msg_ci->p[6]};

    // check for valid intrinsics
    const bool calibrated = msg_ci->width && msg_ci->height &&
                            intrinsics[0] && intrinsics[1] && intrinsics[2] && intrinsics[3];

    if(estimate_pose != nullptr && !calibrated) {
        RCLCPP_WARN_STREAM(get_logger(), "The camera is not calibrated! Set 'pose_estimation_method' to \"\" (empty) to disable pose estimation and this warning.");
    }

    // convert to 8bit monochrome image
    const cv::Mat img_uint8 = cv_bridge::toCvShare(msg_img, "mono8")->image;

    image_u8_t im{img_uint8.cols, img_uint8.rows, img_uint8.cols, img_uint8.data};

    // detect tags
    mutex.lock();
    zarray_t* detections = apriltag_detector_detect(td, &im);
    mutex.unlock();

    if(profile)
        timeprofile_display(td->tp);

    apriltag_msgs::msg::AprilTagDetectionArray msg_detections;
    msg_detections.header = msg_img->header;

    std::vector<geometry_msgs::msg::TransformStamped> tfs;
    std::unordered_map<std::string, std::vector<apriltag_detection_t*>> bundle_detections;

    for(int i = 0; i < zarray_size(detections); i++) {
        apriltag_detection_t* det;
        zarray_get(detections, i, &det);

        RCLCPP_DEBUG(get_logger(),
                     "detection %3d: id (%2dx%2d)-%-4d, hamming %d, margin %8.3f\n",
                     i, det->family->nbits, det->family->h, det->id,
                     det->hamming, det->decision_margin);

        // reject detections with more corrected bits than allowed
        if(det->hamming > max_hamming) { continue; }

        // Define a safety margin in pixels from the image boundary
        const double edge_margin = 50.0;

        // Get image dimensions from the incoming image properties
        const double img_width = img_uint8.cols;
        const double img_height = img_uint8.rows;

        bool near_edge = false;
        for(int corner_idx = 0; corner_idx < 4; corner_idx++) {
            double cx = det->p[corner_idx][0];
            double cy = det->p[corner_idx][1];

            if(cx < edge_margin || cx > (img_width - edge_margin) ||
               cy < edge_margin || cy > (img_height - edge_margin)) {
                near_edge = true;
                break;
            }
        }

        if(near_edge) {
            RCLCPP_DEBUG(get_logger(), "Rejecting tag %d due to edge proximity.", det->id);
            continue;
        }

        // For all detections, extract relevant ones to bundle_detections
        if(tag_id_to_bundles.count(det->id)) {
            bool include_in_bundle = true;

            // Reject tags that are too far from the camera to be reliable bundle correspondences
            const double max_dist = max_tag_distance;
            if(max_dist > 0.0 && estimate_pose != nullptr && calibrated) {
                const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
                const geometry_msgs::msg::Transform tag_transform = estimate_pose(det, intrinsics, size);
                const double distance = std::sqrt(
                    tag_transform.translation.x * tag_transform.translation.x +
                    tag_transform.translation.y * tag_transform.translation.y +
                    tag_transform.translation.z * tag_transform.translation.z);

                if(distance > max_dist) {
                    include_in_bundle = false;
                    RCLCPP_DEBUG(get_logger(), "Excluding tag %d from bundle: estimated distance %.2fm exceeds max_tag_distance %.2fm", det->id, distance, max_dist);
                }
            }

            if(include_in_bundle) {
                for(const TagBundlePtr& bundle : tag_id_to_bundles[det->id]) {
                    bundle_detections[bundle->frame_id].push_back(det);
                }
            }
        }

        // ignore untracked tags
        if(!tag_frames.empty() && !tag_frames.count(det->id)) { continue; }

        // detection
        apriltag_msgs::msg::AprilTagDetection msg_detection;
        msg_detection.family = std::string(det->family->name);
        msg_detection.id = det->id;
        msg_detection.hamming = det->hamming;
        msg_detection.decision_margin = det->decision_margin;
        msg_detection.centre.x = det->c[0];
        msg_detection.centre.y = det->c[1];
        std::memcpy(msg_detection.corners.data(), det->p, sizeof(double) * 8);
        std::memcpy(msg_detection.homography.data(), det->H->data, sizeof(double) * 9);
        msg_detections.detections.push_back(msg_detection);

        // 3D orientation and position
        if(estimate_pose != nullptr && calibrated) {
            geometry_msgs::msg::TransformStamped tf;
            tf.header = msg_img->header;
            // if tag id is already in a bundle and tag is not defined in parameter file, don't estimate and publish transform
            if((tag_id_to_bundles.find(det->id) != tag_id_to_bundles.end()) && (tag_frames.count(det->id) == 0)) { continue; }
            // set child frame name by generic tag name or configured tag name
            tf.child_frame_id = tag_frames.count(det->id) ? tag_frames.at(det->id) : std::string(det->family->name) + ":" + std::to_string(det->id);
            const double size = tag_sizes.count(det->id) ? tag_sizes.at(det->id) : tag_edge_size;
            tf.transform = estimate_pose(det, intrinsics, size);
            tfs.push_back(tf);
        }
    }


    pub_detections->publish(msg_detections);

    if(estimate_pose != nullptr) {
        for(auto& bundle : all_tag_bundles) {
            if(bundle_detections.count(bundle->frame_id)) {
                const PoseEstimate estimate = pnp_bundle(
                    bundle_detections[bundle->frame_id], intrinsics, bundle->id_to_size,
                    bundle->id_to_tf, covariance_pixel_stddev, covariance_scale,
                    covariance_max_condition_number, max_reprojection_error,
                    max_tag_reprojection_error);
                if(!estimate.valid) {
                    RCLCPP_DEBUG(
                        get_logger(), "PnP failed or rejected for bundle %s (RMS %.2f px)",
                        bundle->frame_id.c_str(), estimate.reprojection_error);
                    continue;
                }

                if(estimate.rejected_tag_id >= 0) {
                    RCLCPP_DEBUG(
                        get_logger(), "Bundle %s excluded tag %d and recovered with RMS %.2f px",
                        bundle->frame_id.c_str(), estimate.rejected_tag_id,
                        estimate.reprojection_error);
                }

                geometry_msgs::msg::TransformStamped bundle_transform_stamped;
                bundle_transform_stamped.header = msg_img->header;
                bundle_transform_stamped.child_frame_id = bundle->frame_id;
                bundle_transform_stamped.transform = estimate.transform;
                tf_broadcaster.sendTransform(bundle_transform_stamped);

                if(estimate.covariance_valid) {
                    geometry_msgs::msg::PoseWithCovarianceStamped pose;
                    pose.header = msg_img->header;
                    pose.pose.pose.position.x = estimate.transform.translation.x;
                    pose.pose.pose.position.y = estimate.transform.translation.y;
                    pose.pose.pose.position.z = estimate.transform.translation.z;
                    pose.pose.pose.orientation = estimate.transform.rotation;
                    pose.pose.covariance = estimate.covariance;
                    bundle->pose_publisher->publish(pose);
                }
                else {
                    RCLCPP_DEBUG(
                        get_logger(), "Covariance rejected for ill-conditioned bundle %s",
                        bundle->frame_id.c_str());
                }
            }
        }
        tf_broadcaster.sendTransform(tfs);
    }

    apriltag_detections_destroy(detections);
}

rcl_interfaces::msg::SetParametersResult
AprilTagNode::onParameter(const std::vector<rclcpp::Parameter>& parameters)
{
    rcl_interfaces::msg::SetParametersResult result;

    mutex.lock();

    for(const rclcpp::Parameter& parameter : parameters) {
        RCLCPP_DEBUG_STREAM(get_logger(), "setting: " << parameter);

        IF("detector.threads", td->nthreads)
        IF("detector.decimate", td->quad_decimate)
        IF("detector.blur", td->quad_sigma)
        IF("detector.refine", td->refine_edges)
        IF("detector.sharpening", td->decode_sharpening)
        IF("detector.debug", td->debug)
        IF("max_hamming", max_hamming)
        IF("profile", profile)
        IF("max_tag_distance", max_tag_distance)
        IF("covariance.pixel_stddev", covariance_pixel_stddev)
        IF("covariance.scale", covariance_scale)
        IF("covariance.max_condition_number", covariance_max_condition_number)
        IF("max_reprojection_error", max_reprojection_error)
        IF("max_tag_reprojection_error", max_tag_reprojection_error)
    }

    mutex.unlock();

    result.successful = true;

    return result;
}
