// camera_reconstruction_sift_essential.cpp
//
// Minimal multi-view reconstruction demo using OpenCV:
// - SIFT feature matching between consecutive images
// - Essential matrix estimation (known intrinsics K)
// - Camera pose estimation (R, t) per view
// - 3D point triangulation from inlier correspondences
//
// Build (example):
//   g++ camera_reconstruction_sift_essential.cpp -o recon `pkg-config --cflags --libs opencv4`
//
// Run:
//   ./recon img1.jpg img2.jpg img3.jpg ...
//
// Note: This is a research-style demo, not production code. No bundle adjustment.

#include "extract_images.hpp"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace fs = std::filesystem;

// === CONFIGURATION ===
std::string VIDEO_FILENAME = "IMG_0284.MOV";

// Keyframe selection thresholds
const double MIN_MEDIAN_DISPLACEMENT = 20.0;   // Minimum median pixel displacement for keyframe
const double MAX_MEDIAN_DISPLACEMENT = 150.0;  // Maximum displacement (too much motion = blur)
const int MIN_TRACKED_FEATURES = 100;          // Minimum features to track
const double MIN_INLIER_RATIO = 0.3;           // Minimum inlier ratio for good pose
const int MIN_INLIERS_FOR_KEYFRAME = 50;       // Minimum inliers to accept keyframe

// Triangulation quality thresholds
const double MIN_PARALLAX_DEG = 1.0;           // Minimum triangulation angle (degrees)
const double MAX_REPROJ_ERROR = 4.0;           // Maximum reprojection error for triangulated point
const double MIN_DEPTH = 0.1;                  // Minimum depth (relative to baseline)
const double MAX_DEPTH = 50.0;                 // Maximum depth (relative to baseline)

// Outlier removal thresholds
const double OUTLIER_REPROJ_THRESHOLD = 5.0;   // Remove points with reproj error > this after BA

// Bundle adjustment settings
enum class BAMethod {
    ALTERNATING,       // Alternating optimization (cameras then points) - fast but suboptimal
    JOINT              // Joint optimization of all parameters - slower but better convergence
};
const BAMethod BA_METHOD = BAMethod::JOINT;
const bool USE_HUBER_LOSS = true;              // Use Huber robust loss to handle outliers
const double HUBER_DELTA = 2.0;                // Huber threshold in pixels
const int BA_MAX_ITERATIONS = 30;              // Maximum BA iterations

// Pose graph optimization
enum class PoseGraphMethod {
    SIMPLE_LINEAR,     // Simple linear interpolation of rotation error
    GAUSS_NEWTON       // Full Gauss-Newton optimization on pose graph
};
const PoseGraphMethod POSE_GRAPH_METHOD = PoseGraphMethod::GAUSS_NEWTON;
const int POSE_GRAPH_ITERATIONS = 20;          // Number of GN iterations for pose graph optimization

// Loop closure settings
const bool ENABLE_LOOP_CLOSURE = true;         // Set to false to disable loop closure detection

// Output settings
const bool SAVE_SCENE_POINTS = true;           // Set to false to only save camera trajectory (no 3D points)


struct CameraPose
{
    cv::Mat R;  // 3x3, double
    cv::Mat t;  // 3x1, double
};

struct Observation
{
    int camIndex;        // index of camera (0..numViews-1)
    int pointIndex;      // index of 3D point in all3DPoints
    cv::Point2d pixel;   // observed pixel coordinates (u,v)
};

/**
 * @brief Edge in the pose graph, representing a relative transformation constraint.
 */
struct PoseEdge
{
    int from;           // Index of source pose
    int to;             // Index of target pose
    cv::Mat R_rel;      // Relative rotation: R_to = R_rel * R_from
    cv::Mat t_rel;      // Relative translation: t_to = R_rel * t_from + t_rel
    double weight;      // Weight/confidence of this edge (higher = more trusted)
    bool isLoopClosure; // True if this is a loop closure edge
};

// --- Utility functions --------------------------------------------------------

int extract_images(std::string video_filename) {
    std::string video_name = video_filename.substr(0, video_filename.find('.'));
    std::string output_dir = "data/extracted_frames/" + video_name;
    if (!fs::exists("data") && fs::exists("../data")) {
        output_dir = "../data/extracted_frames/" + video_name;
    }

    if (fs::exists(output_dir)) {
        std::cout << "Output directory " << output_dir << " already exists. Skipping extraction." << std::endl;
        return 0;
    }

    std::string video_path = "data/" + video_filename;
    if (!fs::exists(video_path)) {
        video_path = "../data/" + video_filename;
    }
    
    std::cout << "Starting image extraction from: " << video_path << std::endl;
    if (extract_images_from_mov(video_path, output_dir)) {
        std::cout << "Extraction completed successfully." << std::endl;
    } else {
        std::cerr << "Extraction failed." << std::endl;
        return -1;
    }

    return 0;
}

// Convert CameraPose (R, t) to 6D parameter vector p = [rx, ry, rz, tx, ty, tz]
// where (rx, ry, rz) is Rodrigues rotation vector (angle-axis).
inline void poseToParams(const CameraPose& pose, double p[6])
{
    cv::Mat rvec;
    cv::Rodrigues(pose.R, rvec);

    p[0] = rvec.at<double>(0);
    p[1] = rvec.at<double>(1);
    p[2] = rvec.at<double>(2);
    p[3] = pose.t.at<double>(0);
    p[4] = pose.t.at<double>(1);
    p[5] = pose.t.at<double>(2);
}

// Convert 6D parameters back to CameraPose.
inline void paramsToPose(const double p[6], CameraPose& pose)
{
    cv::Mat rvec(3, 1, CV_64F);
    rvec.at<double>(0) = p[0];
    rvec.at<double>(1) = p[1];
    rvec.at<double>(2) = p[2];

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    pose.R = R.clone();
    pose.t = (cv::Mat_<double>(3, 1) << p[3], p[4], p[5]);
}

// Project a 3D point using intrinsics K and pose (R,t).
inline cv::Point2d projectPoint(
    const cv::Mat& K,
    const CameraPose& pose,
    const cv::Point3d& X)
{
    cv::Mat Xw = (cv::Mat_<double>(3,1) << X.x, X.y, X.z);
    cv::Mat Xc = pose.R * Xw + pose.t; // camera coordinates

    double x = Xc.at<double>(0);
    double y = Xc.at<double>(1);
    double z = Xc.at<double>(2);

    double u = K.at<double>(0,0) * x / z + K.at<double>(0,2);
    double v = K.at<double>(1,1) * y / z + K.at<double>(1,2);

    return cv::Point2d(u, v);
}

/**
 * @brief Compute median displacement between matched point pairs.
 * Used for keyframe selection - ensures sufficient parallax.
 */
double computeMedianDisplacement(
    const std::vector<cv::Point2f>& pts1,
    const std::vector<cv::Point2f>& pts2)
{
    if (pts1.empty()) return 0.0;
    
    std::vector<double> displacements;
    displacements.reserve(pts1.size());
    
    for (size_t i = 0; i < pts1.size(); ++i)
    {
        double dx = pts2[i].x - pts1[i].x;
        double dy = pts2[i].y - pts1[i].y;
        displacements.push_back(std::sqrt(dx*dx + dy*dy));
    }
    
    std::sort(displacements.begin(), displacements.end());
    return displacements[displacements.size() / 2];
}

/**
 * @brief Compute the triangulation angle (parallax) between two camera rays.
 * Good triangulation requires sufficient parallax (typically > 1-2 degrees).
 * 
 * @param C1 Camera center 1 in world coordinates
 * @param C2 Camera center 2 in world coordinates
 * @param X  3D point in world coordinates
 * @return Angle in degrees between the two rays
 */
double computeParallaxAngle(
    const cv::Mat& C1,
    const cv::Mat& C2,
    const cv::Point3d& X)
{
    cv::Mat Xm = (cv::Mat_<double>(3,1) << X.x, X.y, X.z);
    
    cv::Mat ray1 = Xm - C1;
    cv::Mat ray2 = Xm - C2;
    
    double norm1 = cv::norm(ray1);
    double norm2 = cv::norm(ray2);
    
    if (norm1 < 1e-9 || norm2 < 1e-9) return 0.0;
    
    ray1 /= norm1;
    ray2 /= norm2;
    
    double cosAngle = ray1.dot(ray2);
    cosAngle = std::max(-1.0, std::min(1.0, cosAngle)); // Clamp for numerical stability
    
    return std::acos(cosAngle) * 180.0 / CV_PI;
}

/**
 * @brief Compute reprojection error for a single 3D point in one camera.
 */
double computeSingleReprojError(
    const cv::Mat& K,
    const CameraPose& pose,
    const cv::Point3d& X,
    const cv::Point2d& observed)
{
    cv::Mat Xw = (cv::Mat_<double>(3,1) << X.x, X.y, X.z);
    cv::Mat Xc = pose.R * Xw + pose.t;
    
    double z = Xc.at<double>(2);
    if (z <= 0) return 1e9; // Behind camera
    
    double u = K.at<double>(0,0) * Xc.at<double>(0) / z + K.at<double>(0,2);
    double v = K.at<double>(1,1) * Xc.at<double>(1) / z + K.at<double>(1,2);
    
    double dx = u - observed.x;
    double dy = v - observed.y;
    
    return std::sqrt(dx*dx + dy*dy);
}

/**
 * @brief Compute median of a vector of doubles.
 */
double computeMedian(std::vector<double>& values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

/**
 * @brief Huber weight for robust optimization (IRLS).
 * 
 * Returns the weight to apply to a residual for Huber loss.
 * Used in iteratively reweighted least squares.
 * 
 * For |r| <= delta: weight = 1.0 (quadratic region)
 * For |r| > delta:  weight = delta / |r| (linear region)
 */
inline double huberWeight(double residual, double delta)
{
    double absRes = std::abs(residual);
    if (absRes <= delta)
        return 1.0;
    else
        return delta / absRes;
}

/**
 * @brief Compute the rotation error between two rotations in axis-angle representation.
 * Returns the angle in radians.
 */
double rotationError(const cv::Mat& R1, const cv::Mat& R2)
{
    cv::Mat R_diff = R1 * R2.t();
    cv::Mat rvec;
    cv::Rodrigues(R_diff, rvec);
    return cv::norm(rvec);
}

/**
 * @brief Pose Graph Optimizer using Gauss-Newton.
 * 
 * Optimizes camera poses given a set of relative pose constraints (edges).
 * The first pose is held fixed as the reference frame.
 * 
 * Each pose is parameterized as 6 DoF: [rx, ry, rz, tx, ty, tz]
 * where (rx, ry, rz) is the Rodrigues rotation vector.
 * 
 * The cost function minimizes:
 *   sum over edges: weight * ||log(R_rel^T * R_to * R_from^T)||^2 + weight * ||t_error||^2
 */
void optimizePoseGraph(
    std::vector<CameraPose>& poses,
    const std::vector<PoseEdge>& edges,
    int maxIterations = 20)
{
    const int numPoses = static_cast<int>(poses.size());
    if (numPoses < 2 || edges.empty()) return;
    
    // 6 DoF per pose, but pose 0 is fixed
    const int numParams = (numPoses - 1) * 6;
    
    std::cout << "  Pose graph optimization: " << numPoses << " poses, " 
              << edges.size() << " edges" << std::endl;
    
    // Convert poses to parameter vector (skip pose 0 which is fixed)
    auto posesToParams = [&](std::vector<double>& params) {
        params.resize(numParams);
        for (int i = 1; i < numPoses; ++i) {
            if (poses[i].R.empty()) {
                // Invalid pose - initialize to identity
                for (int j = 0; j < 6; ++j) params[(i-1)*6 + j] = 0.0;
            } else {
                cv::Mat rvec;
                cv::Rodrigues(poses[i].R, rvec);
                params[(i-1)*6 + 0] = rvec.at<double>(0);
                params[(i-1)*6 + 1] = rvec.at<double>(1);
                params[(i-1)*6 + 2] = rvec.at<double>(2);
                params[(i-1)*6 + 3] = poses[i].t.at<double>(0);
                params[(i-1)*6 + 4] = poses[i].t.at<double>(1);
                params[(i-1)*6 + 5] = poses[i].t.at<double>(2);
            }
        }
    };
    
    auto paramsToPoses = [&](const std::vector<double>& params) {
        for (int i = 1; i < numPoses; ++i) {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 
                params[(i-1)*6 + 0], params[(i-1)*6 + 1], params[(i-1)*6 + 2]);
            cv::Rodrigues(rvec, poses[i].R);
            poses[i].t = (cv::Mat_<double>(3,1) << 
                params[(i-1)*6 + 3], params[(i-1)*6 + 4], params[(i-1)*6 + 5]);
        }
    };
    
    // Get pose R and t from params (handling pose 0 specially)
    auto getPose = [&](const std::vector<double>& params, int idx, cv::Mat& R, cv::Mat& t) {
        if (idx == 0) {
            R = poses[0].R.clone();
            t = poses[0].t.clone();
        } else {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 
                params[(idx-1)*6 + 0], params[(idx-1)*6 + 1], params[(idx-1)*6 + 2]);
            cv::Rodrigues(rvec, R);
            t = (cv::Mat_<double>(3,1) << 
                params[(idx-1)*6 + 3], params[(idx-1)*6 + 4], params[(idx-1)*6 + 5]);
        }
    };
    
    std::vector<double> params;
    posesToParams(params);
    
    const double eps = 1e-6;
    
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Compute residuals and Jacobian numerically
        const int numResiduals = static_cast<int>(edges.size()) * 6; // 3 for rotation, 3 for translation per edge
        
        cv::Mat residuals(numResiduals, 1, CV_64F);
        cv::Mat J(numResiduals, numParams, CV_64F, cv::Scalar(0));
        
        // Compute residuals for current params
        auto computeResiduals = [&](const std::vector<double>& p, cv::Mat& r) {
            int rIdx = 0;
            for (const auto& edge : edges)
            {
                cv::Mat R_from, t_from, R_to, t_to;
                getPose(p, edge.from, R_from, t_from);
                getPose(p, edge.to, R_to, t_to);
                
                // Expected: R_to = R_rel * R_from, t_to = R_rel * t_from + t_rel
                // Error rotation: R_err = R_rel^T * R_to * R_from^T
                cv::Mat R_expected = edge.R_rel * R_from;
                cv::Mat R_err = R_expected.t() * R_to;
                
                cv::Mat rvec_err;
                cv::Rodrigues(R_err, rvec_err);
                
                // Error translation
                cv::Mat t_expected = edge.R_rel * t_from + edge.t_rel;
                cv::Mat t_err = t_to - t_expected;
                
                double w = std::sqrt(edge.weight);
                
                r.at<double>(rIdx + 0) = w * rvec_err.at<double>(0);
                r.at<double>(rIdx + 1) = w * rvec_err.at<double>(1);
                r.at<double>(rIdx + 2) = w * rvec_err.at<double>(2);
                r.at<double>(rIdx + 3) = w * t_err.at<double>(0);
                r.at<double>(rIdx + 4) = w * t_err.at<double>(1);
                r.at<double>(rIdx + 5) = w * t_err.at<double>(2);
                
                rIdx += 6;
            }
        };
        
        computeResiduals(params, residuals);
        
        double cost = residuals.dot(residuals);
        
        // Numeric Jacobian
        for (int j = 0; j < numParams; ++j)
        {
            std::vector<double> params_plus = params;
            std::vector<double> params_minus = params;
            params_plus[j] += eps;
            params_minus[j] -= eps;
            
            cv::Mat r_plus(numResiduals, 1, CV_64F);
            cv::Mat r_minus(numResiduals, 1, CV_64F);
            
            computeResiduals(params_plus, r_plus);
            computeResiduals(params_minus, r_minus);
            
            cv::Mat col = (r_plus - r_minus) / (2 * eps);
            col.copyTo(J.col(j));
        }
        
        // Gauss-Newton: H * dp = -g where H = J^T * J, g = J^T * r
        cv::Mat H = J.t() * J;
        cv::Mat g = J.t() * residuals;
        
        // Add damping (Levenberg-Marquardt style)
        double lambda = 1e-4 * cv::trace(H)[0] / numParams;
        for (int k = 0; k < numParams; ++k)
            H.at<double>(k, k) += lambda;
        
        cv::Mat dp;
        bool ok = cv::solve(H, -g, dp, cv::DECOMP_CHOLESKY);
        if (!ok) {
            std::cout << "  PGO iteration " << iter << ": solve failed" << std::endl;
            break;
        }
        
        // Update params
        double maxUpdate = 0;
        for (int k = 0; k < numParams; ++k) {
            params[k] += dp.at<double>(k);
            maxUpdate = std::max(maxUpdate, std::abs(dp.at<double>(k)));
        }
        
        if (iter % 5 == 0 || iter == maxIterations - 1) {
            std::cout << "  PGO iteration " << iter << ": cost=" << std::scientific 
                      << cost << ", maxUpdate=" << maxUpdate << std::endl;
        }
        
        if (maxUpdate < 1e-6) {
            std::cout << "  PGO converged at iteration " << iter << std::endl;
            break;
        }
    }
    
    // Write back optimized poses
    paramsToPoses(params);
}

/**
 * @brief Simple linear pose correction (original method).
 * Distributes rotation error linearly along the trajectory.
 */
void simplePoseCorrection(
    std::vector<CameraPose>& poses,
    int loopFrameIdx,
    int pastFrameIdx,
    const cv::Mat& R_loop,
    const cv::Mat& /* t_loop - unused due to scale ambiguity */)
{
    cv::Mat R_past = poses[pastFrameIdx].R;
    cv::Mat t_past = poses[pastFrameIdx].t;
    cv::Mat R_curr = poses[loopFrameIdx].R;
    
    // Sequential relative pose
    cv::Mat R_seq = R_curr * R_past.t();
    
    // Rotation error
    cv::Mat R_err = R_loop * R_seq.t();
    cv::Mat rvec_err;
    cv::Rodrigues(R_err, rvec_err);
    
    double angleErr = cv::norm(rvec_err);
    std::cout << "  Rotation drift: " << angleErr * 180.0 / CV_PI << " degrees" << std::endl;
    
    // Distribute error linearly
    int numFramesToCorrect = loopFrameIdx - pastFrameIdx;
    std::cout << "  Correcting poses for frames " << (pastFrameIdx + 1) 
              << " to " << loopFrameIdx << std::endl;
    
    for (int f = pastFrameIdx + 1; f <= loopFrameIdx; ++f)
    {
        if (poses[f].R.empty()) continue;
        
        double alpha = (double)(f - pastFrameIdx) / numFramesToCorrect;
        
        cv::Mat rvec_correction = alpha * rvec_err;
        cv::Mat R_correction;
        cv::Rodrigues(rvec_correction, R_correction);
        
        poses[f].R = R_correction * poses[f].R;
    }
    
    std::cout << "  Simple pose correction applied." << std::endl;
}

/**
 * @brief Estimate camera pose from 2D-3D correspondences using PnP.
 * 
 * This gives us the pose directly in the map coordinate system,
 * avoiding scale drift that occurs with essential matrix estimation.
 * PnP maintains consistent scale because we're localizing against
 * existing 3D points that already have proper scale.
 * 
 * @param K Camera intrinsic matrix.
 * @param points2D 2D image points (in current frame).
 * @param points3D Corresponding 3D world points (from existing map).
 * @param R Output rotation matrix (world to camera).
 * @param t Output translation vector (world to camera).
 * @param inliers Output indices of inlier correspondences.
 * @return true if pose estimation succeeded with enough inliers.
 */
bool estimatePoseFromPnP(
    const cv::Mat& K,
    const std::vector<cv::Point2f>& points2D,
    const std::vector<cv::Point3d>& points3D,
    cv::Mat& R,
    cv::Mat& t,
    std::vector<int>& inliers)
{
    const int MIN_PNP_INLIERS = 15;
    
    if (points2D.size() < 6 || points3D.size() < 6) {
        return false;
    }
    
    if (points2D.size() != points3D.size()) {
        std::cerr << "PnP: mismatched 2D/3D point counts" << std::endl;
        return false;
    }
    
    cv::Mat rvec, tvec;
    
    bool success = cv::solvePnPRansac(
        points3D,
        points2D,
        K,
        cv::noArray(),  // No distortion (images already undistorted)
        rvec,
        tvec,
        false,          // useExtrinsicGuess
        300,            // iterationsCount
        4.0,            // reprojectionError threshold (pixels)
        0.99,           // confidence
        inliers,
        cv::SOLVEPNP_ITERATIVE
    );
    
    if (!success || static_cast<int>(inliers.size()) < MIN_PNP_INLIERS) {
        return false;
    }
    
    // Refine with iterative PnP using only inliers
    std::vector<cv::Point2f> inlierPts2D;
    std::vector<cv::Point3d> inlierPts3D;
    inlierPts2D.reserve(inliers.size());
    inlierPts3D.reserve(inliers.size());
    
    for (int idx : inliers) {
        inlierPts2D.push_back(points2D[idx]);
        inlierPts3D.push_back(points3D[idx]);
    }
    
    // Refine the pose estimate
    cv::solvePnP(
        inlierPts3D,
        inlierPts2D,
        K,
        cv::noArray(),
        rvec,
        tvec,
        true,  // useExtrinsicGuess - refine the RANSAC result
        cv::SOLVEPNP_ITERATIVE
    );
    
    // Convert rvec to rotation matrix
    cv::Rodrigues(rvec, R);
    t = tvec.clone();
    
    // Ensure proper types
    R.convertTo(R, CV_64F);
    t.convertTo(t, CV_64F);
    
    return true;
}

/**
 * @brief Detect SIFT features and compute descriptors for an image.
 */
void detectAndDescribeSIFT(
    const cv::Mat &image,
    std::vector<cv::KeyPoint> &keypoints,
    cv::Mat &descriptors)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(4000);
    sift->detectAndCompute(image, cv::noArray(), keypoints, descriptors);
}

/**
 * @brief Match SIFT descriptors between two images using Lowe's ratio test.
 */
void matchFeatures(
    const cv::Mat &desc1,
    const cv::Mat &desc2,
    std::vector<cv::DMatch> &goodMatches,
    double ratio = 0.75)
{
    goodMatches.clear();

    cv::BFMatcher matcher(cv::NORM_L2, false); // SIFT -> L2, no crossCheck for ratio test
    std::vector<std::vector<cv::DMatch>> knnMatches;
    matcher.knnMatch(desc1, desc2, knnMatches, 2);

    for (const auto &m : knnMatches)
    {
        if (m.size() < 2)
            continue;

        const cv::DMatch &best = m[0];
        const cv::DMatch &second = m[1];

        if (best.distance < ratio * second.distance)
        {
            goodMatches.push_back(best);
        }
    }
}

/**
 * @brief Extract Point2f correspondences from keypoints and matches.
 */
void extractMatchedPoints(
    const std::vector<cv::KeyPoint> &kpts1,
    const std::vector<cv::KeyPoint> &kpts2,
    const std::vector<cv::DMatch> &matches,
    std::vector<cv::Point2f> &pts1,
    std::vector<cv::Point2f> &pts2)
{
    pts1.clear();
    pts2.clear();
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());

    for (const auto &m : matches)
    {
        pts1.push_back(kpts1[m.queryIdx].pt);
        pts2.push_back(kpts2[m.trainIdx].pt);
    }
}

/**
 * @brief Estimate relative pose (R, t) from matched points using essential matrix.
 * 
 * @param K Camera intrinsic matrix (3x3, double).
 * @param pts1 Points in image 1 (pixel coordinates).
 * @param pts2 Corresponding points in image 2 (pixel coordinates).
 * @param R Output rotation from cam1 to cam2.
 * @param t Output translation from cam1 to cam2 (up to scale).
 * @param inlierMask Output inlier mask from recoverPose (uchar).
 */
bool estimateRelativePoseFromEssential(
    const cv::Mat &K,
    const std::vector<cv::Point2f> &pts1,
    const std::vector<cv::Point2f> &pts2,
    cv::Mat &R,
    cv::Mat &t,
    cv::Mat &inlierMask)
{
    if (pts1.size() < 8 || pts2.size() < 8)
    {
        std::cerr << "Not enough point correspondences (need >= 8)." << std::endl;
        return false;
    }

    // Find essential matrix with RANSAC
    double focal = K.at<double>(0, 0);
    cv::Point2d pp(K.at<double>(0, 2), K.at<double>(1, 2));

    cv::Mat E = cv::findEssentialMat(
        pts1, pts2,
        K,
        cv::RANSAC,
        0.999,
        1.0,
        inlierMask
    );

    if (E.empty())
    {
        std::cerr << "Essential matrix estimation failed." << std::endl;
        return false;
    }

    int inliers = cv::recoverPose(
        E,
        pts1,
        pts2,
        K,
        R,
        t,
        inlierMask
    );

    if (inliers < 10)
    {
        std::cerr << "Too few inliers after recoverPose: " << inliers << std::endl;
        return false;
    }

    return true;
}

/**
 * @brief Refine a single camera pose given fixed 3D points.
 *
 * Uses simple Gauss-Newton with numeric Jacobian and optional Huber loss.
 *
 * @param K Intrinsic matrix.
 * @param camIndex Index of the camera to refine.
 * @param poses Vector of camera poses (will be updated at camIndex).
 * @param points3D All 3D points (fixed).
 * @param observations All observations.
 * @param maxIters Number of Gauss-Newton iterations.
 * @param useHuber Whether to use Huber robust loss.
 * @param huberDelta Huber threshold in pixels.
 */
void refineCameraPoseGN(
    const cv::Mat& K,
    int camIndex,
    std::vector<CameraPose>& poses,
    const std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations,
    int maxIters = 10,
    bool useHuber = false,
    double huberDelta = 2.0)
{
    // Collect observations for this camera
    std::vector<const Observation*> obsForCam;
    obsForCam.reserve(observations.size());
    for (const auto& obs : observations)
    {
        if (obs.camIndex == camIndex)
            obsForCam.push_back(&obs);
    }

    if (obsForCam.size() < 10)
        return; // too few observations to be worth refining

    CameraPose& pose = poses[camIndex];
    double p[6];
    poseToParams(pose, p);

    const double eps = 1e-6;

    for (int iter = 0; iter < maxIters; ++iter)
    {
        // Build residual vector r and compute Huber weights
        const int m = static_cast<int>(obsForCam.size());
        cv::Mat r(2*m, 1, CV_64F);
        std::vector<double> weights(m, 1.0);

        for (int i = 0; i < m; ++i)
        {
            const Observation* ob = obsForCam[i];
            const cv::Point3d& X = points3D[ob->pointIndex];

            CameraPose tmpPose;
            paramsToPose(p, tmpPose);
            cv::Point2d proj = projectPoint(K, tmpPose, X);

            double rx = proj.x - ob->pixel.x;
            double ry = proj.y - ob->pixel.y;
            r.at<double>(2*i+0) = rx;
            r.at<double>(2*i+1) = ry;
            
            // Compute Huber weight if enabled
            if (useHuber)
            {
                double err = std::sqrt(rx*rx + ry*ry);
                weights[i] = huberWeight(err, huberDelta);
            }
        }

        // Numeric Jacobian J (2m x 6)
        cv::Mat J(2*m, 6, CV_64F);

        for (int k = 0; k < 6; ++k)
        {
            double p_plus[6], p_minus[6];
            for (int j = 0; j < 6; ++j)
            {
                p_plus[j]  = p[j];
                p_minus[j] = p[j];
            }
            p_plus[k]  += eps;
            p_minus[k] -= eps;

            cv::Mat r_plus(2*m, 1, CV_64F);
            cv::Mat r_minus(2*m, 1, CV_64F);

            for (int i = 0; i < m; ++i)
            {
                const Observation* ob = obsForCam[i];
                const cv::Point3d& X = points3D[ob->pointIndex];

                CameraPose posePlus, poseMinus;
                paramsToPose(p_plus, posePlus);
                paramsToPose(p_minus, poseMinus);

                cv::Point2d projPlus  = projectPoint(K, posePlus,  X);
                cv::Point2d projMinus = projectPoint(K, poseMinus, X);

                r_plus.at<double>(2*i+0) = projPlus.x  - ob->pixel.x;
                r_plus.at<double>(2*i+1) = projPlus.y  - ob->pixel.y;
                r_minus.at<double>(2*i+0) = projMinus.x - ob->pixel.x;
                r_minus.at<double>(2*i+1) = projMinus.y - ob->pixel.y;
            }

            cv::Mat col = (r_plus - r_minus) * (0.5 / eps);
            col.copyTo(J.col(k));
        }

        // Apply Huber weights to Jacobian and residuals (IRLS)
        if (useHuber)
        {
            for (int i = 0; i < m; ++i)
            {
                double w = std::sqrt(weights[i]);
                J.row(2*i) *= w;
                J.row(2*i + 1) *= w;
                r.at<double>(2*i) *= w;
                r.at<double>(2*i + 1) *= w;
            }
        }

        // Normal equations: H * dp = -g
        cv::Mat H = J.t() * J;
        cv::Mat g = J.t() * r;

        // Simple damping (Levenberg-like)
        double lambda = 1e-3;
        for (int k = 0; k < 6; ++k)
            H.at<double>(k,k) += lambda;

        cv::Mat dp;
        bool ok = cv::solve(H, -g, dp, cv::DECOMP_CHOLESKY);
        if (!ok)
            break;

        double maxUpdate = 0.0;
        for (int k = 0; k < 6; ++k)
        {
            p[k] += dp.at<double>(k);
            maxUpdate = std::max(maxUpdate, std::abs(dp.at<double>(k)));
        }

        if (maxUpdate < 1e-6)
            break; // converged
    }

    // write back refined pose
    paramsToPose(p, pose);
}

/**
 * @brief Refine a single 3D point given fixed camera poses.
 *
 * Uses simple Gauss-Newton with numeric Jacobian and optional Huber loss.
 *
 * @param K Intrinsic matrix.
 * @param pointIndex Index of the 3D point to refine.
 * @param poses All camera poses (fixed).
 * @param points3D Vector of 3D points (will be updated at pointIndex).
 * @param observations All observations.
 * @param maxIters Number of Gauss-Newton iterations.
 * @param useHuber Whether to use Huber robust loss.
 * @param huberDelta Huber threshold in pixels.
 */
void refinePointGN(
    const cv::Mat& K,
    int pointIndex,
    const std::vector<CameraPose>& poses,
    std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations,
    int maxIters = 10,
    bool useHuber = false,
    double huberDelta = 2.0)
{
    // Collect observations seeing this point
    std::vector<const Observation*> obsForPoint;
    obsForPoint.reserve(observations.size());
    for (const auto& obs : observations)
    {
        if (obs.pointIndex == pointIndex)
            obsForPoint.push_back(&obs);
    }

    if (obsForPoint.size() < 2)
        return; // at least 2 views needed

    cv::Point3d& X = points3D[pointIndex];
    double p[3] = { X.x, X.y, X.z };

    const double eps = 1e-6;

    for (int iter = 0; iter < maxIters; ++iter)
    {
        int m = static_cast<int>(obsForPoint.size());
        cv::Mat r(2*m, 1, CV_64F);
        std::vector<double> weights(m, 1.0);

        for (int i = 0; i < m; ++i)
        {
            const Observation* ob = obsForPoint[i];
            const CameraPose& pose = poses[ob->camIndex];

            cv::Point3d Xcur(p[0], p[1], p[2]);
            cv::Point2d proj = projectPoint(K, pose, Xcur);

            double rx = proj.x - ob->pixel.x;
            double ry = proj.y - ob->pixel.y;
            r.at<double>(2*i+0) = rx;
            r.at<double>(2*i+1) = ry;
            
            // Compute Huber weight if enabled
            if (useHuber)
            {
                double err = std::sqrt(rx*rx + ry*ry);
                weights[i] = huberWeight(err, huberDelta);
            }
        }

        cv::Mat J(2*m, 3, CV_64F);

        for (int k = 0; k < 3; ++k)
        {
            double p_plus[3]  = { p[0], p[1], p[2] };
            double p_minus[3] = { p[0], p[1], p[2] };
            p_plus[k]  += eps;
            p_minus[k] -= eps;

            cv::Mat r_plus(2*m, 1, CV_64F);
            cv::Mat r_minus(2*m, 1, CV_64F);

            for (int i = 0; i < m; ++i)
            {
                const Observation* ob = obsForPoint[i];
                const CameraPose& pose = poses[ob->camIndex];

                cv::Point3d Xp(p_plus[0],  p_plus[1],  p_plus[2]);
                cv::Point3d Xm(p_minus[0], p_minus[1], p_minus[2]);

                cv::Point2d projPlus  = projectPoint(K, pose, Xp);
                cv::Point2d projMinus = projectPoint(K, pose, Xm);

                r_plus.at<double>(2*i+0)  = projPlus.x  - ob->pixel.x;
                r_plus.at<double>(2*i+1)  = projPlus.y  - ob->pixel.y;
                r_minus.at<double>(2*i+0) = projMinus.x - ob->pixel.x;
                r_minus.at<double>(2*i+1) = projMinus.y - ob->pixel.y;
            }

            cv::Mat col = (r_plus - r_minus) * (0.5 / eps);
            col.copyTo(J.col(k));
        }

        // Apply Huber weights to Jacobian and residuals (IRLS)
        if (useHuber)
        {
            for (int i = 0; i < m; ++i)
            {
                double w = std::sqrt(weights[i]);
                J.row(2*i) *= w;
                J.row(2*i + 1) *= w;
                r.at<double>(2*i) *= w;
                r.at<double>(2*i + 1) *= w;
            }
        }

        cv::Mat H = J.t() * J;
        cv::Mat g = J.t() * r;

        double lambda = 1e-3;
        for (int k = 0; k < 3; ++k)
            H.at<double>(k,k) += lambda;

        cv::Mat dp;
        bool ok = cv::solve(H, -g, dp, cv::DECOMP_CHOLESKY);
        if (!ok)
            break;

        double maxUpdate = 0.0;
        for (int k = 0; k < 3; ++k)
        {
            p[k] += dp.at<double>(k);
            maxUpdate = std::max(maxUpdate, std::abs(dp.at<double>(k)));
        }

        if (maxUpdate < 1e-6)
            break;
    }

    X.x = p[0];
    X.y = p[1];
    X.z = p[2];
}


/**
 * @brief Compute reprojection error for all observations.
 *
 * @param K             Camera intrinsic matrix.
 * @param poses         Vector of camera poses.
 * @param points3D      Vector of 3D points.
 * @param observations  List of 2D observations.
 *
 * @return Mean pixel error across all observations.
 */
double computeReprojectionError(
    const cv::Mat& K,
    const std::vector<CameraPose>& poses,
    const std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations)
{
    double totalError = 0.0;
    int count = 0;

    for (const auto& obs : observations)
    {
        const CameraPose& pose = poses[obs.camIndex];
        const cv::Point3d& X   = points3D[obs.pointIndex];
        cv::Point2d proj = projectPoint(K, pose, X);

        double dx = proj.x - obs.pixel.x;
        double dy = proj.y - obs.pixel.y;
        double err = std::sqrt(dx*dx + dy*dy);

        totalError += err;
        count++;
    }

    if (count == 0) return 0.0;
    return totalError / count;
}

/**
 * @brief Joint Bundle Adjustment with Levenberg-Marquardt and optional Huber loss.
 * 
 * Jointly optimizes all camera poses and 3D points to minimize reprojection error.
 * Camera 0 is held fixed as the reference frame.
 * 
 * Parameter layout:
 *   [cam1_rx, cam1_ry, cam1_rz, cam1_tx, cam1_ty, cam1_tz,
 *    cam2_rx, ..., camN_rx, ...,
 *    pt0_x, pt0_y, pt0_z, pt1_x, ...]
 * 
 * Uses numeric Jacobian for simplicity. For production code, use Ceres or g2o.
 * 
 * @param K Camera intrinsic matrix.
 * @param poses Vector of camera poses (will be updated, pose 0 is fixed).
 * @param points3D Vector of 3D points (will be updated).
 * @param observations List of 2D observations.
 * @param useHuber Whether to use Huber robust loss.
 * @param huberDelta Huber threshold in pixels.
 * @param maxIterations Maximum LM iterations.
 * @return true if optimization ran, false if problem was too large (caller should fall back)
 */
bool jointBundleAdjustment(
    const cv::Mat& K,
    std::vector<CameraPose>& poses,
    std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations,
    bool useHuber = true,
    double huberDelta = 2.0,
    int maxIterations = 30)
{
    if (poses.empty() || points3D.empty() || observations.empty())
    {
        std::cerr << "Nothing to optimize in BA." << std::endl;
        return true;  // Nothing to do, but not a fallback case
    }

    const int numCams = static_cast<int>(poses.size());
    const int numPts = static_cast<int>(points3D.size());
    const int numObs = static_cast<int>(observations.size());
    
    // Parameter counts (camera 0 is fixed)
    const int camParamCount = (numCams - 1) * 6;
    const int ptParamCount = numPts * 3;
    const int totalParams = camParamCount + ptParamCount;
    const int totalResiduals = numObs * 2;  // 2 residuals (u, v) per observation
    
    // Check problem size - dense BA requires storing full Jacobian matrix
    // Memory needed: totalResiduals * totalParams * 8 bytes
    // For safety, limit to problems where Jacobian < ~1GB
    const long long jacobianSize = static_cast<long long>(totalResiduals) * totalParams;
    const long long MAX_JACOBIAN_ELEMENTS = 100000000LL;  // ~800MB for doubles
    
    if (jacobianSize > MAX_JACOBIAN_ELEMENTS)
    {
        std::cout << "\nProblem too large for dense joint BA:" << std::endl;
        std::cout << "  Jacobian would be " << totalResiduals << " x " << totalParams 
                  << " = " << (jacobianSize * 8 / 1e9) << " GB" << std::endl;
        std::cout << "  Falling back to alternating BA..." << std::endl;
        return false;  // Signal caller to use alternating BA
    }
    
    std::cout << "\nStarting joint bundle adjustment:" << std::endl;
    std::cout << "  Cameras: " << numCams << " (" << camParamCount << " params, cam 0 fixed)" << std::endl;
    std::cout << "  Points: " << numPts << " (" << ptParamCount << " params)" << std::endl;
    std::cout << "  Observations: " << numObs << " (" << totalResiduals << " residuals)" << std::endl;
    std::cout << "  Huber loss: " << (useHuber ? "enabled" : "disabled");
    if (useHuber) std::cout << " (delta=" << huberDelta << " px)";
    std::cout << std::endl;
    
    // === Convert current state to parameter vector ===
    std::vector<double> params(totalParams);
    
    // Pack camera parameters (skip camera 0)
    for (int c = 1; c < numCams; ++c)
    {
        if (poses[c].R.empty()) {
            // Initialize invalid poses to identity
            for (int j = 0; j < 6; ++j) 
                params[(c-1)*6 + j] = 0.0;
            continue;
        }
        cv::Mat rvec;
        cv::Rodrigues(poses[c].R, rvec);
        params[(c-1)*6 + 0] = rvec.at<double>(0);
        params[(c-1)*6 + 1] = rvec.at<double>(1);
        params[(c-1)*6 + 2] = rvec.at<double>(2);
        params[(c-1)*6 + 3] = poses[c].t.at<double>(0);
        params[(c-1)*6 + 4] = poses[c].t.at<double>(1);
        params[(c-1)*6 + 5] = poses[c].t.at<double>(2);
    }
    
    // Pack point parameters
    for (int p = 0; p < numPts; ++p)
    {
        params[camParamCount + p*3 + 0] = points3D[p].x;
        params[camParamCount + p*3 + 1] = points3D[p].y;
        params[camParamCount + p*3 + 2] = points3D[p].z;
    }
    
    // === Helper to unpack parameters ===
    auto unpackParams = [&](const std::vector<double>& p) {
        // Unpack cameras
        for (int c = 1; c < numCams; ++c)
        {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 
                p[(c-1)*6 + 0], p[(c-1)*6 + 1], p[(c-1)*6 + 2]);
            cv::Rodrigues(rvec, poses[c].R);
            poses[c].t = (cv::Mat_<double>(3,1) << 
                p[(c-1)*6 + 3], p[(c-1)*6 + 4], p[(c-1)*6 + 5]);
        }
        // Unpack points
        for (int pt = 0; pt < numPts; ++pt)
        {
            points3D[pt].x = p[camParamCount + pt*3 + 0];
            points3D[pt].y = p[camParamCount + pt*3 + 1];
            points3D[pt].z = p[camParamCount + pt*3 + 2];
        }
    };
    
    // === Helper to compute residuals and optionally Jacobian ===
    auto computeResidualsAndJacobian = [&](
        const std::vector<double>& p, 
        cv::Mat& residuals,
        cv::Mat* J,  // nullptr if Jacobian not needed
        std::vector<double>* weights  // Huber weights, nullptr if not needed
    ) {
        // Temporarily unpack parameters to compute projections
        // (We need poses and points in their native form for projectPoint)
        std::vector<CameraPose> tempPoses = poses;
        std::vector<cv::Point3d> tempPoints = points3D;
        
        // Unpack cameras
        for (int c = 1; c < numCams; ++c)
        {
            cv::Mat rvec = (cv::Mat_<double>(3,1) << 
                p[(c-1)*6 + 0], p[(c-1)*6 + 1], p[(c-1)*6 + 2]);
            cv::Rodrigues(rvec, tempPoses[c].R);
            tempPoses[c].t = (cv::Mat_<double>(3,1) << 
                p[(c-1)*6 + 3], p[(c-1)*6 + 4], p[(c-1)*6 + 5]);
        }
        // Unpack points
        for (int pt = 0; pt < numPts; ++pt)
        {
            tempPoints[pt].x = p[camParamCount + pt*3 + 0];
            tempPoints[pt].y = p[camParamCount + pt*3 + 1];
            tempPoints[pt].z = p[camParamCount + pt*3 + 2];
        }
        
        residuals = cv::Mat(totalResiduals, 1, CV_64F);
        if (weights) weights->resize(numObs);
        
        for (int i = 0; i < numObs; ++i)
        {
            const Observation& obs = observations[i];
            const CameraPose& pose = tempPoses[obs.camIndex];
            const cv::Point3d& X = tempPoints[obs.pointIndex];
            
            cv::Point2d proj = projectPoint(K, pose, X);
            
            double rx = proj.x - obs.pixel.x;
            double ry = proj.y - obs.pixel.y;
            
            residuals.at<double>(2*i + 0) = rx;
            residuals.at<double>(2*i + 1) = ry;
            
            if (weights)
            {
                double err = std::sqrt(rx*rx + ry*ry);
                (*weights)[i] = useHuber ? huberWeight(err, huberDelta) : 1.0;
            }
        }
        
        // Compute Jacobian numerically if requested
        if (J)
        {
            const double eps = 1e-6;
            *J = cv::Mat(totalResiduals, totalParams, CV_64F, cv::Scalar(0));
            
            // For each observation, only compute derivatives w.r.t. relevant parameters
            // (the camera that sees it and the point being observed)
            for (int i = 0; i < numObs; ++i)
            {
                const Observation& obs = observations[i];
                int camIdx = obs.camIndex;
                int ptIdx = obs.pointIndex;
                
                // Camera parameters (skip if camera 0)
                if (camIdx > 0)
                {
                    int camParamStart = (camIdx - 1) * 6;
                    
                    for (int k = 0; k < 6; ++k)
                    {
                        std::vector<double> p_plus = p;
                        std::vector<double> p_minus = p;
                        p_plus[camParamStart + k] += eps;
                        p_minus[camParamStart + k] -= eps;
                        
                        // Recompute projection for perturbed camera
                        cv::Mat rvec_p = (cv::Mat_<double>(3,1) << 
                            p_plus[(camIdx-1)*6 + 0], p_plus[(camIdx-1)*6 + 1], p_plus[(camIdx-1)*6 + 2]);
                        cv::Mat R_p;
                        cv::Rodrigues(rvec_p, R_p);
                        cv::Mat t_p = (cv::Mat_<double>(3,1) << 
                            p_plus[(camIdx-1)*6 + 3], p_plus[(camIdx-1)*6 + 4], p_plus[(camIdx-1)*6 + 5]);
                        CameraPose pose_p; pose_p.R = R_p; pose_p.t = t_p;
                        cv::Point2d proj_p = projectPoint(K, pose_p, tempPoints[ptIdx]);
                        
                        cv::Mat rvec_m = (cv::Mat_<double>(3,1) << 
                            p_minus[(camIdx-1)*6 + 0], p_minus[(camIdx-1)*6 + 1], p_minus[(camIdx-1)*6 + 2]);
                        cv::Mat R_m;
                        cv::Rodrigues(rvec_m, R_m);
                        cv::Mat t_m = (cv::Mat_<double>(3,1) << 
                            p_minus[(camIdx-1)*6 + 3], p_minus[(camIdx-1)*6 + 4], p_minus[(camIdx-1)*6 + 5]);
                        CameraPose pose_m; pose_m.R = R_m; pose_m.t = t_m;
                        cv::Point2d proj_m = projectPoint(K, pose_m, tempPoints[ptIdx]);
                        
                        J->at<double>(2*i + 0, camParamStart + k) = (proj_p.x - proj_m.x) / (2 * eps);
                        J->at<double>(2*i + 1, camParamStart + k) = (proj_p.y - proj_m.y) / (2 * eps);
                    }
                }
                
                // Point parameters
                int ptParamStart = camParamCount + ptIdx * 3;
                
                for (int k = 0; k < 3; ++k)
                {
                    cv::Point3d X_p = tempPoints[ptIdx];
                    cv::Point3d X_m = tempPoints[ptIdx];
                    
                    if (k == 0) { X_p.x += eps; X_m.x -= eps; }
                    else if (k == 1) { X_p.y += eps; X_m.y -= eps; }
                    else { X_p.z += eps; X_m.z -= eps; }
                    
                    cv::Point2d proj_p = projectPoint(K, tempPoses[camIdx], X_p);
                    cv::Point2d proj_m = projectPoint(K, tempPoses[camIdx], X_m);
                    
                    J->at<double>(2*i + 0, ptParamStart + k) = (proj_p.x - proj_m.x) / (2 * eps);
                    J->at<double>(2*i + 1, ptParamStart + k) = (proj_p.y - proj_m.y) / (2 * eps);
                }
            }
        }
    };
    
    // === Levenberg-Marquardt optimization ===
    double lambda = 1e-3;  // LM damping parameter
    const double lambdaUp = 10.0;
    const double lambdaDown = 0.1;
    
    cv::Mat residuals;
    std::vector<double> weights;
    computeResidualsAndJacobian(params, residuals, nullptr, &weights);
    
    double prevCost = 0;
    for (int i = 0; i < numObs; ++i)
    {
        double rx = residuals.at<double>(2*i);
        double ry = residuals.at<double>(2*i + 1);
        double w = weights[i];
        prevCost += w * (rx*rx + ry*ry);
    }
    
    for (int iter = 0; iter < maxIterations; ++iter)
    {
        // Compute Jacobian
        cv::Mat J;
        computeResidualsAndJacobian(params, residuals, &J, &weights);
        
        // Apply Huber weights to Jacobian and residuals
        if (useHuber)
        {
            for (int i = 0; i < numObs; ++i)
            {
                double w = std::sqrt(weights[i]);
                J.row(2*i) *= w;
                J.row(2*i + 1) *= w;
                residuals.at<double>(2*i) *= w;
                residuals.at<double>(2*i + 1) *= w;
            }
        }
        
        // Normal equations: H * dp = -g where H = J^T * J, g = J^T * r
        cv::Mat H = J.t() * J;
        cv::Mat g = J.t() * residuals;
        
        // Apply LM damping
        for (int k = 0; k < totalParams; ++k)
        {
            H.at<double>(k, k) *= (1.0 + lambda);
        }
        
        // Solve for update
        cv::Mat dp;
        bool solved = cv::solve(H, -g, dp, cv::DECOMP_CHOLESKY);
        
        if (!solved)
        {
            // Try with more regularization
            lambda *= lambdaUp;
            if (lambda > 1e10) {
                std::cout << "  LM failed to converge (solve failed)" << std::endl;
                break;
            }
            continue;
        }
        
        // Trial update
        std::vector<double> paramsNew(totalParams);
        for (int k = 0; k < totalParams; ++k)
        {
            paramsNew[k] = params[k] + dp.at<double>(k);
        }
        
        // Compute new cost
        cv::Mat residualsNew;
        std::vector<double> weightsNew;
        computeResidualsAndJacobian(paramsNew, residualsNew, nullptr, &weightsNew);
        
        double newCost = 0;
        for (int i = 0; i < numObs; ++i)
        {
            double rx = residualsNew.at<double>(2*i);
            double ry = residualsNew.at<double>(2*i + 1);
            double w = weightsNew[i];
            newCost += w * (rx*rx + ry*ry);
        }
        
        if (newCost < prevCost)
        {
            // Accept update
            params = paramsNew;
            lambda *= lambdaDown;
            lambda = std::max(lambda, 1e-10);
            
            double improvement = (prevCost - newCost) / prevCost;
            prevCost = newCost;
            
            // Compute current reprojection error
            unpackParams(params);
            double meanErr = computeReprojectionError(K, poses, points3D, observations);
            
            if (iter % 5 == 0 || iter == maxIterations - 1)
            {
                std::cout << "  Iter " << std::setw(2) << iter 
                          << ": cost=" << std::scientific << std::setprecision(3) << newCost
                          << ", err=" << std::fixed << std::setprecision(2) << meanErr << " px"
                          << ", lambda=" << std::scientific << std::setprecision(1) << lambda
                          << std::endl;
            }
            
            // Check convergence
            if (improvement < 1e-6)
            {
                std::cout << "  Converged at iteration " << iter << " (cost improvement < 1e-6)" << std::endl;
                break;
            }
        }
        else
        {
            // Reject update, increase damping
            lambda *= lambdaUp;
            if (lambda > 1e10) {
                std::cout << "  LM converged at iteration " << iter << " (lambda overflow)" << std::endl;
                break;
            }
        }
    }
    
    // Final unpack
    unpackParams(params);
    
    double finalErr = computeReprojectionError(K, poses, points3D, observations);
    std::cout << "Joint bundle adjustment finished. Final error: " << finalErr << " px" << std::endl;
    return true;
}

/**
 * @brief Alternating bundle adjustment:
 *        - refine all camera poses with fixed 3D points
 *        - refine all 3D points with fixed camera poses
 *
 * Repeat for a few outer iterations.
 */
void alternatingBundleAdjustment(
    const cv::Mat& K,
    std::vector<CameraPose>& poses,
    std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations,
    int numOuterIters = 5,
    bool useHuber = false,
    double huberDelta = 2.0)
{
    if (poses.empty() || points3D.empty() || observations.empty())
    {
        std::cerr << "Nothing to optimize in BA." << std::endl;
        return;
    }

    std::cout << "\nStarting alternating bundle adjustment with "
              << numOuterIters << " outer iterations";
    if (useHuber)
        std::cout << " (Huber loss, delta=" << huberDelta << " px)";
    std::cout << "..." << std::endl;

    for (int iter = 0; iter < numOuterIters; ++iter)
    {
        std::cout << "  BA outer iteration " << iter+1 << " / "
                  << numOuterIters;

        // Pose-only step
        for (int camIdx = 0; camIdx < static_cast<int>(poses.size()); ++camIdx)
        {
            refineCameraPoseGN(K, camIdx, poses, points3D, observations, 5, useHuber, huberDelta);
        }

        // Point-only step
        for (int ptIdx = 0; ptIdx < static_cast<int>(points3D.size()); ++ptIdx)
        {
            refinePointGN(K, ptIdx, poses, points3D, observations, 5, useHuber, huberDelta);
        }

        double err = computeReprojectionError(K, poses, points3D, observations);
        std::cout << ": " << err << " px.\n";
    }

    std::cout << "Alternating bundle adjustment finished.\n";
}

/**
 * @brief Run bundle adjustment using the configured method.
 * 
 * Dispatches to either joint or alternating BA based on BA_METHOD setting.
 */
void runBundleAdjustment(
    const cv::Mat& K,
    std::vector<CameraPose>& poses,
    std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations,
    int numIterations = 5)
{
    if (BA_METHOD == BAMethod::JOINT)
    {
        // Try joint BA, fall back to alternating if problem is too large
        bool success = jointBundleAdjustment(K, poses, points3D, observations, 
                                             USE_HUBER_LOSS, HUBER_DELTA, BA_MAX_ITERATIONS);
        if (!success)
        {
            // Problem too large for dense joint BA, use alternating instead with Huber
            alternatingBundleAdjustment(K, poses, points3D, observations, numIterations,
                                        USE_HUBER_LOSS, HUBER_DELTA);
        }
    }
    else
    {
        alternatingBundleAdjustment(K, poses, points3D, observations, numIterations,
                                    USE_HUBER_LOSS, HUBER_DELTA);
    }
}

/**
 * @brief Save 3D points and camera positions into a Wavefront OBJ file.
 *
 * MeshLab can directly import this file.
 *
 * @param filename Path to .obj file to write.
 * @param points3D Vector of 3D points (Point3d).
 * @param poses Vector of camera poses (R,t), world coordinate system.
 *
 * Notes:
 *  - Cameras are exported as vertices as well.
 *  - Camera centers are computed as: C = -R^T * t
 *  - No faces are generated because this is a point cloud + cameras only.
 */
void saveAsOBJ(
    const std::string &filename,
    const std::vector<cv::Point3d> &points3D,
    const std::vector<CameraPose> &poses,
    bool savePoints = true)
{
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "Could not write OBJ file: " << filename << std::endl;
        return;
    }

    out << "# OBJ file generated by OpenCV reconstruction demo\n";
    out << "# Number of 3D points: " << (savePoints ? points3D.size() : 0) << "\n";
    out << "# Number of cameras: " << poses.size() << "\n";
    out << "# Scene points saved: " << (savePoints ? "yes" : "no") << "\n\n";

    // --- Write 3D points (if enabled) ----------------------------------------
    if (savePoints)
    {
        out << "# 3D points\n";
        for (const auto &p : points3D)
        {
            out << "v " << p.x << " " << p.y << " " << p.z << "\n";
        }
    }

    out << "\n# Camera centers\n";

    // --- Write camera centers -----------------------------------------------
    // Camera center C = -R^T * t
    for (size_t i = 0; i < poses.size(); ++i)
    {
        if (poses[i].R.empty() || poses[i].t.empty())
        {
            std::cerr << "Warning: Skipping camera " << i << " (invalid pose) in OBJ export." << std::endl;
            continue;
        }

        const cv::Mat R = poses[i].R;
        const cv::Mat t = poses[i].t;

        cv::Mat C = -R.t() * t;  // camera center in world coordinates

        out << "v " << C.at<double>(0) << " "
                    << C.at<double>(1) << " "
                    << C.at<double>(2) << "\n";
    }

    // (Optional) Write camera axes for visualization
    out << "\n# Camera axes (Optional small lines)\n";
    for (size_t i = 0; i < poses.size(); ++i)
    {
        if (poses[i].R.empty() || poses[i].t.empty()) continue;

        const cv::Mat R = poses[i].R;
        const cv::Mat t = poses[i].t;

        cv::Mat C = -R.t() * t;

        // axis length
        double s = 0.1;

        cv::Vec3d X = cv::Vec3d(R.at<double>(0,0), R.at<double>(1,0), R.at<double>(2,0));
        cv::Vec3d Y = cv::Vec3d(R.at<double>(0,1), R.at<double>(1,1), R.at<double>(2,1));
        cv::Vec3d Z = cv::Vec3d(R.at<double>(0,2), R.at<double>(1,2), R.at<double>(2,2));

        cv::Point3d Cw(C.at<double>(0), C.at<double>(1), C.at<double>(2));

        cv::Point3d Xend = Cw + s * cv::Point3d(X);
        cv::Point3d Yend = Cw + s * cv::Point3d(Y);
        cv::Point3d Zend = Cw + s * cv::Point3d(Z);

        // Export as vertices (MeshLab can show them)
        out << "v " << Xend.x << " " << Xend.y << " " << Xend.z << "\n";
        out << "v " << Yend.x << " " << Yend.y << " " << Yend.z << "\n";
        out << "v " << Zend.x << " " << Zend.y << " " << Zend.z << "\n";
    }

    out.close();
    std::cout << "OBJ written to: " << filename << std::endl;
}


// --- Main --------------------------------------------------------------------

int main(int argc, char** argv)
{
    try
    {
        std::string video_name = VIDEO_FILENAME.substr(0, VIDEO_FILENAME.find('.'));

        extract_images(VIDEO_FILENAME);

        // Find the extracted frames directory
        std::string extracted_frames_dir = "data/extracted_frames/" + video_name;
        if (!fs::exists(extracted_frames_dir)) {
            extracted_frames_dir = "../data/extracted_frames/" + video_name;
            if (!fs::exists(extracted_frames_dir)) {
                std::cerr << "Could not find extracted frames directory: " << extracted_frames_dir << std::endl;
                return -1;
            }
        }

        // find ALL frames (we'll select keyframes dynamically)
        std::vector<std::string> allFramePaths;
        for (int i = 0; ; ++i) {
            std::string frame_path = extracted_frames_dir + "/frame_" + std::format("{:04d}", i) + ".png";
            if (!fs::exists(frame_path)) break;
            allFramePaths.push_back(frame_path);
        }
        
        std::cout << "Found " << allFramePaths.size() << " total frames in sequence." << std::endl;

        // --- 1. Camera intrinsics (K) ----------------------------------------
        //
        // TODO: Replace with your actual calibration matrix. Values below are
        // a typical example (fx, fy, cx, cy) for a ~640x480 camera.
        //
        // K must be double-precision for the code below.
        // Camera intrinsics + distortion from chessboard calibration
        const double fx = 1226.991674550505;
        const double fy = 1231.583548480416;
        const double cx = 529.5391035340654;
        const double cy = 936.7114915473007;
        cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5)
            << 0.009593106889362086, -0.08836017837645339,
            -0.002369764239215277, -0.002095085353035259,
            0.1736273482549004);

        

        // --- 2. Keyframe-based processing with dynamic selection ---------------
        //
        // Instead of loading all frames, we process incrementally:
        // - Load frame, extract features
        // - Match to last keyframe
        // - Decide if this frame should be a keyframe based on:
        //   1) Sufficient parallax (median displacement)
        //   2) Enough tracked features
        //   3) Good geometric consistency
        
        std::vector<cv::Mat> keyframeImages;
        std::vector<std::vector<cv::KeyPoint>> allKeypoints;
        std::vector<cv::Mat> allDescriptors;
        std::vector<CameraPose> poses;
        std::vector<int> keyframeIndices; // Maps keyframe index to original frame index
        
        std::vector<cv::Point3d> all3DPoints;
        std::vector<Observation> observations;
        
        // Track which keypoint in which keyframe corresponds to which 3D point
        std::vector<std::vector<int>> keypointToPointIdx;
        
        // Process first frame as keyframe 0
        {
            cv::Mat img = cv::imread(allFramePaths[0], cv::IMREAD_GRAYSCALE);
            cv::Mat imgUndistorted;
            cv::undistort(img, imgUndistorted, K, distCoeffs);
            keyframeImages.push_back(imgUndistorted);
            
            std::vector<cv::KeyPoint> kpts;
            cv::Mat desc;
            detectAndDescribeSIFT(imgUndistorted, kpts, desc);
            allKeypoints.push_back(kpts);
            allDescriptors.push_back(desc);
            
            CameraPose pose0;
            pose0.R = cv::Mat::eye(3, 3, CV_64F);
            pose0.t = cv::Mat::zeros(3, 1, CV_64F);
            poses.push_back(pose0);
            
            keypointToPointIdx.push_back(std::vector<int>(kpts.size(), -1));
            keyframeIndices.push_back(0);
            
            std::cout << "Keyframe 0 (frame 0): " << kpts.size() << " keypoints" << std::endl;
        }
        
        int lastKeyframeIdx = 0;  // Index in keyframe arrays
        int lastKeyframeFrameIdx = 0;  // Index in original frame sequence
        
        // Process remaining frames
        for (size_t frameIdx = 1; frameIdx < allFramePaths.size(); ++frameIdx)
        {
            // Load and undistort current frame
            cv::Mat img = cv::imread(allFramePaths[frameIdx], cv::IMREAD_GRAYSCALE);
            if (img.empty()) continue;
            
            cv::Mat imgUndistorted;
            cv::undistort(img, imgUndistorted, K, distCoeffs);
            
            // Extract features
            std::vector<cv::KeyPoint> currKpts;
            cv::Mat currDesc;
            detectAndDescribeSIFT(imgUndistorted, currKpts, currDesc);
            
            // Match to last keyframe
            std::vector<cv::DMatch> matches;
            matchFeatures(allDescriptors[lastKeyframeIdx], currDesc, matches);
            
            if (matches.size() < MIN_TRACKED_FEATURES)
            {
                // Too few matches - might need to force a keyframe or skip
                continue;
            }
            
            // Extract matched points
            std::vector<cv::Point2f> pts1, pts2;
            extractMatchedPoints(allKeypoints[lastKeyframeIdx], currKpts, matches, pts1, pts2);
            
            // === KEYFRAME SELECTION CRITERIA ===
            double medianDisp = computeMedianDisplacement(pts1, pts2);
            
            // Check if we have sufficient parallax
            if (medianDisp < MIN_MEDIAN_DISPLACEMENT)
            {
                // Not enough motion yet - skip this frame
                continue;
            }
            
            if (medianDisp > MAX_MEDIAN_DISPLACEMENT)
            {
                // Too much motion - might be blurry or tracking lost
                std::cout << "Frame " << frameIdx << ": displacement too large (" 
                          << medianDisp << " px), skipping" << std::endl;
                continue;
            }
            
            // === POSE ESTIMATION ===
            // Strategy: Try PnP first (if we have existing 3D map points), 
            // fall back to essential matrix if not enough correspondences.
            // PnP gives us absolute pose with consistent scale.
            // Essential matrix gives relative pose with unit-norm translation (scale drift).
            
            cv::Mat inlierMask, R, t;
            bool usedPnP = false;
            int inlierCount = 0;
            
            // Collect 2D-3D correspondences from existing map points
            std::vector<cv::Point2f> pts2DForPnP;
            std::vector<cv::Point3d> pts3DForPnP;
            std::vector<int> matchIndicesWithMap;  // Which matches have existing 3D points
            
            for (size_t k = 0; k < matches.size(); ++k)
            {
                int kptIdx1 = matches[k].queryIdx;  // Index in last keyframe
                int existingPointIdx = keypointToPointIdx[lastKeyframeIdx][kptIdx1];
                
                if (existingPointIdx != -1 && existingPointIdx < static_cast<int>(all3DPoints.size()))
                {
                    pts2DForPnP.push_back(currKpts[matches[k].trainIdx].pt);
                    pts3DForPnP.push_back(all3DPoints[existingPointIdx]);
                    matchIndicesWithMap.push_back(static_cast<int>(k));
                }
            }
            
            const int MIN_POINTS_FOR_PNP = 25;  // Need enough 2D-3D correspondences
            
            // Try PnP if we have enough map point observations
            if (pts2DForPnP.size() >= MIN_POINTS_FOR_PNP)
            {
                std::vector<int> pnpInliers;
                cv::Mat R_pnp, t_pnp;
                
                if (estimatePoseFromPnP(K, pts2DForPnP, pts3DForPnP, R_pnp, t_pnp, pnpInliers))
                {
                    usedPnP = true;
                    R = R_pnp;
                    t = t_pnp;
                    inlierCount = static_cast<int>(pnpInliers.size());
                    
                    // Create inlier mask for all matches
                    // Mark PnP inliers, and also run essential matrix to identify
                    // additional inliers among matches without existing 3D points
                    inlierMask = cv::Mat::zeros(static_cast<int>(matches.size()), 1, CV_8U);
                    
                    // Mark PnP inliers
                    for (int idx : pnpInliers) {
                        int matchIdx = matchIndicesWithMap[idx];
                        inlierMask.at<uchar>(matchIdx) = 1;
                    }
                    
                    // For matches WITHOUT existing 3D points, verify geometrically
                    // by checking reprojection with the estimated pose
                    // This allows us to triangulate new points from these matches
                    cv::Mat P_new(3, 4, CV_64F);
                    cv::hconcat(R, t, P_new);
                    P_new = K * P_new;
                    
                    cv::Mat P_prev(3, 4, CV_64F);
                    cv::hconcat(poses[lastKeyframeIdx].R, poses[lastKeyframeIdx].t, P_prev);
                    P_prev = K * P_prev;
                    
                    // Compute fundamental matrix from the two projection matrices
                    // F = [e']_x * P' * P^+  (but we'll use epipolar constraint directly)
                    for (size_t k = 0; k < matches.size(); ++k)
                    {
                        if (inlierMask.at<uchar>(static_cast<int>(k))) continue;  // Already marked
                        
                        int kptIdx1 = matches[k].queryIdx;
                        if (keypointToPointIdx[lastKeyframeIdx][kptIdx1] != -1) continue;
                        
                        // Triangulate this point and check reprojection error
                        cv::Mat pts4D;
                        std::vector<cv::Point2f> p1 = {pts1[k]};
                        std::vector<cv::Point2f> p2 = {pts2[k]};
                        cv::triangulatePoints(P_prev, P_new, p1, p2, pts4D);
                        
                        double w = pts4D.at<float>(3, 0);
                        if (std::abs(w) < 1e-9) continue;
                        
                        cv::Point3d X(
                            pts4D.at<float>(0, 0) / w,
                            pts4D.at<float>(1, 0) / w,
                            pts4D.at<float>(2, 0) / w
                        );
                        
                        // Check reprojection in both views
                        double err1 = computeSingleReprojError(K, poses[lastKeyframeIdx], X, 
                                                               cv::Point2d(pts1[k].x, pts1[k].y));
                        
                        CameraPose tempPose;
                        tempPose.R = R;
                        tempPose.t = t;
                        double err2 = computeSingleReprojError(K, tempPose, X, 
                                                               cv::Point2d(pts2[k].x, pts2[k].y));
                        
                        // Check depth positivity
                        cv::Mat Xw = (cv::Mat_<double>(3,1) << X.x, X.y, X.z);
                        cv::Mat Xc1 = poses[lastKeyframeIdx].R * Xw + poses[lastKeyframeIdx].t;
                        cv::Mat Xc2 = R * Xw + t;
                        
                        if (Xc1.at<double>(2) > 0 && Xc2.at<double>(2) > 0 &&
                            err1 < MAX_REPROJ_ERROR && err2 < MAX_REPROJ_ERROR)
                        {
                            inlierMask.at<uchar>(static_cast<int>(k)) = 1;
                            inlierCount++;
                        }
                    }
                }
            }
            
            // Fall back to essential matrix if PnP failed or not enough map points
            if (!usedPnP)
            {
                if (!estimateRelativePoseFromEssential(K, pts1, pts2, R, t, inlierMask))
                {
                    continue;
                }
                inlierCount = cv::countNonZero(inlierMask);
            }
            
            double inlierRatio = (double)inlierCount / matches.size();
            
            if (inlierCount < MIN_INLIERS_FOR_KEYFRAME || inlierRatio < MIN_INLIER_RATIO)
            {
                continue;
            }
            
            // === ACCEPT AS KEYFRAME ===
            int newKeyframeIdx = static_cast<int>(poses.size());
            
            std::cout << "\nKeyframe " << newKeyframeIdx << " (frame " << frameIdx << "): "
                      << "disp=" << std::fixed << std::setprecision(1) << medianDisp << "px, "
                      << "matches=" << matches.size() << ", "
                      << "inliers=" << inlierCount << " (" << std::setprecision(0) << inlierRatio*100 << "%)"
                      << (usedPnP ? " [PnP]" : " [Essential]")
                      << std::endl;
            
            // Store keyframe data
            keyframeImages.push_back(imgUndistorted);
            allKeypoints.push_back(currKpts);
            allDescriptors.push_back(currDesc);
            keyframeIndices.push_back(static_cast<int>(frameIdx));
            keypointToPointIdx.push_back(std::vector<int>(currKpts.size(), -1));
            
            // Compute global pose
            CameraPose newPose;
            if (usedPnP)
            {
                // PnP gives us the ABSOLUTE pose directly (with correct scale!)
                newPose.R = R.clone();
                newPose.t = t.clone();
            }
            else
            {
                // Essential matrix gives RELATIVE pose (unit-norm translation)
                // Must compose with previous pose
                newPose.R = R * poses[lastKeyframeIdx].R;
                newPose.t = R * poses[lastKeyframeIdx].t + t;
            }
            poses.push_back(newPose);
            
            // === TRIANGULATE AND MERGE MAP POINTS ===
            // Compute camera centers for parallax check
            cv::Mat C1 = -poses[lastKeyframeIdx].R.t() * poses[lastKeyframeIdx].t;
            cv::Mat C2 = -newPose.R.t() * newPose.t;
            double baseline = cv::norm(C2 - C1);
            
            // Build projection matrices
            cv::Mat P1(3, 4, CV_64F), P2(3, 4, CV_64F);
            cv::hconcat(poses[lastKeyframeIdx].R, poses[lastKeyframeIdx].t, P1);
            P1 = K * P1;
            cv::hconcat(newPose.R, newPose.t, P2);
            P2 = K * P2;
            
            // Collect inlier points and track original indices
            std::vector<cv::Point2f> inlierPts1, inlierPts2;
            std::vector<int> inlierMatchIndices;
            
            for (size_t k = 0; k < matches.size(); ++k)
            {
                if (inlierMask.at<uchar>(static_cast<int>(k)))
                {
                    inlierPts1.push_back(pts1[k]);
                    inlierPts2.push_back(pts2[k]);
                    inlierMatchIndices.push_back(static_cast<int>(k));
                }
            }
            
            // Triangulate
            cv::Mat points4D;
            cv::triangulatePoints(P1, P2, inlierPts1, inlierPts2, points4D);
            
            int nTriangulated = 0;
            int nMerged = 0;
            int nRejectedParallax = 0;
            int nRejectedReproj = 0;
            int nRejectedDepth = 0;
            
            // Collect depths for scale estimation (only from newly triangulated points)
            std::vector<double> newPointDepths;
            
            for (int k = 0; k < points4D.cols; ++k)
            {
                int matchIdx = inlierMatchIndices[k];
                int kptIdx1 = matches[matchIdx].queryIdx;
                int kptIdx2 = matches[matchIdx].trainIdx;
                
                cv::Mat col = points4D.col(k);
                double w = col.at<float>(3, 0);
                if (std::abs(w) < 1e-9) continue;
                
                double X = col.at<float>(0, 0) / w;
                double Y = col.at<float>(1, 0) / w;
                double Z = col.at<float>(2, 0) / w;
                
                cv::Point3d Xw(X, Y, Z);
                cv::Mat XwMat = (cv::Mat_<double>(3,1) << X, Y, Z);
                
                // Cheirality check - depth in both cameras must be positive
                cv::Mat Xc1 = poses[lastKeyframeIdx].R * XwMat + poses[lastKeyframeIdx].t;
                cv::Mat Xc2 = newPose.R * XwMat + newPose.t;
                double depth1 = Xc1.at<double>(2);
                double depth2 = Xc2.at<double>(2);
                
                if (depth1 <= 0 || depth2 <= 0) {
                    nRejectedDepth++;
                    continue;
                }
                
                // Depth range check (relative to baseline)
                double relativeDepth = depth1 / baseline;
                if (relativeDepth < MIN_DEPTH || relativeDepth > MAX_DEPTH) {
                    nRejectedDepth++;
                    continue;
                }
                
                // Parallax angle check - ensures well-conditioned triangulation
                double parallax = computeParallaxAngle(C1, C2, Xw);
                if (parallax < MIN_PARALLAX_DEG) {
                    nRejectedParallax++;
                    continue;
                }
                
                // Reprojection error check
                cv::Point2d obs1(inlierPts1[k].x, inlierPts1[k].y);
                cv::Point2d obs2(inlierPts2[k].x, inlierPts2[k].y);
                
                double err1 = computeSingleReprojError(K, poses[lastKeyframeIdx], Xw, obs1);
                double err2 = computeSingleReprojError(K, newPose, Xw, obs2);
                
                if (err1 > MAX_REPROJ_ERROR || err2 > MAX_REPROJ_ERROR) {
                    nRejectedReproj++;
                    continue;
                }
                
                // === MAP POINT MERGING ===
                int existingPointIdx = keypointToPointIdx[lastKeyframeIdx][kptIdx1];
                
                if (existingPointIdx != -1)
                {
                    // MERGE: Add observation to existing point
                    observations.push_back({newKeyframeIdx, existingPointIdx, obs2});
                    keypointToPointIdx[newKeyframeIdx][kptIdx2] = existingPointIdx;
                    nMerged++;
                }
                else
                {
                    // NEW POINT: Create new 3D point
                    int pointIndex = static_cast<int>(all3DPoints.size());
                    all3DPoints.push_back(Xw);
                    
                    observations.push_back({lastKeyframeIdx, pointIndex, obs1});
                    observations.push_back({newKeyframeIdx, pointIndex, obs2});
                    
                    keypointToPointIdx[lastKeyframeIdx][kptIdx1] = pointIndex;
                    keypointToPointIdx[newKeyframeIdx][kptIdx2] = pointIndex;
                    nTriangulated++;
                    
                    // Store depth for scale estimation
                    newPointDepths.push_back(depth1);
                }
            }
            
            std::cout << "  New: " << nTriangulated << ", Merged: " << nMerged 
                      << " (rejected: parallax=" << nRejectedParallax 
                      << ", reproj=" << nRejectedReproj 
                      << ", depth=" << nRejectedDepth << ")" << std::endl;
            
            // Update last keyframe
            lastKeyframeIdx = newKeyframeIdx;
            lastKeyframeFrameIdx = static_cast<int>(frameIdx);
        }
        
        const int numViews = static_cast<int>(poses.size());
        std::cout << "\n=== Keyframe Selection Complete ===" << std::endl;
        std::cout << "Total keyframes: " << numViews << " (from " << allFramePaths.size() << " frames)" << std::endl;
        std::cout << "Total 3D points: " << all3DPoints.size() << std::endl;

        // --- 3. Loop Closure ------------------------------------------------
        // Find the SINGLE BEST loop closure across the entire trajectory
        // Then correct pose drift before adding observations
        
        if (ENABLE_LOOP_CLOSURE)
        {
            std::cout << "\n=== Starting Loop Closure Detection ===" << std::endl;
            
            int loopGap = std::max(3, numViews / 2); // Loop must span at least half the trajectory
            
            // Variables to track the globally best loop closure
            int globalBestCurrFrame = -1;
            int globalBestPastFrame = -1;
            int globalMaxInliers = -1;
            std::vector<cv::DMatch> globalBestMatches;
            cv::Mat globalBestMask;
            cv::Mat globalBestR, globalBestT;
            
            // Search for the single best loop closure
            for (int curr = loopGap; curr < numViews; ++curr)
            {
                if (poses[curr].R.empty()) continue;
                
                for (int past = 0; past <= curr - loopGap; ++past)
                {
                    if (poses[past].R.empty()) continue;
                    if (allDescriptors[curr].rows < 100 || allDescriptors[past].rows < 100) continue;
                    
                    // Match features with stricter ratio test
                    std::vector<cv::DMatch> matches;
                    matchFeatures(allDescriptors[curr], allDescriptors[past], matches, 0.7);
                    
                    if (matches.size() < 300) continue; // High threshold for loop closure
                    
                    // Geometric verification
                    std::vector<cv::Point2f> ptsCurr, ptsPast;
                    extractMatchedPoints(allKeypoints[curr], allKeypoints[past], matches, ptsCurr, ptsPast);
                    
                    cv::Mat mask;
                    cv::Mat E = cv::findEssentialMat(ptsCurr, ptsPast, K, cv::RANSAC, 0.999, 1.0, mask);
                    
                    if (E.empty()) continue;
                    
                    int inliers = cv::countNonZero(mask);
                    double inlierRatio = (double)inliers / matches.size();
                    
                    // Very strict criteria for loop closure
                    if (inliers > 200 && inlierRatio > 0.6 && inliers > globalMaxInliers)
                    {
                        // Recover relative pose
                        cv::Mat R_loop, t_loop;
                        int poseInliers = cv::recoverPose(E, ptsCurr, ptsPast, K, R_loop, t_loop, mask);
                        
                        if (poseInliers > 100)
                        {
                            globalMaxInliers = inliers;
                            globalBestCurrFrame = curr;
                            globalBestPastFrame = past;
                            globalBestMatches = matches;
                            globalBestMask = mask.clone();
                            globalBestR = R_loop.clone();
                            globalBestT = t_loop.clone();
                        }
                    }
                }
            }
            
            if (globalBestCurrFrame != -1)
            {
                std::cout << "  Best loop closure: Frame " << globalBestCurrFrame 
                          << " <-> Frame " << globalBestPastFrame 
                          << " (" << globalMaxInliers << " inliers)" << std::endl;
                
                // === POSE GRAPH OPTIMIZATION ===
                if (POSE_GRAPH_METHOD == PoseGraphMethod::SIMPLE_LINEAR)
                {
                    std::cout << "  Using simple linear pose correction..." << std::endl;
                    simplePoseCorrection(poses, globalBestCurrFrame, globalBestPastFrame, 
                                         globalBestR, globalBestT);
                }
                else // GAUSS_NEWTON
                {
                    std::cout << "  Using Gauss-Newton pose graph optimization..." << std::endl;
                    
                    // Build pose graph with sequential edges
                    std::vector<PoseEdge> poseEdges;
                    
                    // Add sequential edges (odometry constraints)
                    for (int i = 1; i < numViews; ++i)
                    {
                        if (poses[i-1].R.empty() || poses[i].R.empty()) continue;
                        
                        // Relative pose: R_i = R_rel * R_{i-1}, t_i = R_rel * t_{i-1} + t_rel
                        cv::Mat R_rel = poses[i].R * poses[i-1].R.t();
                        cv::Mat t_rel = poses[i].t - R_rel * poses[i-1].t;
                        
                        PoseEdge edge;
                        edge.from = i - 1;
                        edge.to = i;
                        edge.R_rel = R_rel.clone();
                        edge.t_rel = t_rel.clone();
                        edge.weight = 1.0;  // Sequential edges have unit weight
                        edge.isLoopClosure = false;
                        poseEdges.push_back(edge);
                    }
                    
                    // Add loop closure edge
                    PoseEdge loopEdge;
                    loopEdge.from = globalBestPastFrame;
                    loopEdge.to = globalBestCurrFrame;
                    loopEdge.R_rel = globalBestR.clone();
                    loopEdge.t_rel = globalBestT.clone();
                    loopEdge.weight = 10.0;  // Loop closures have higher weight (more trusted)
                    loopEdge.isLoopClosure = true;
                    poseEdges.push_back(loopEdge);
                    
                    std::cout << "  Built pose graph: " << poseEdges.size() << " edges ("
                              << (poseEdges.size() - 1) << " sequential + 1 loop closure)" << std::endl;
                    
                    // Compute rotation drift before optimization
                    cv::Mat R_seq = poses[globalBestCurrFrame].R * poses[globalBestPastFrame].R.t();
                    cv::Mat R_err = globalBestR * R_seq.t();
                    cv::Mat rvec_err;
                    cv::Rodrigues(R_err, rvec_err);
                    double angleErr = cv::norm(rvec_err);
                    std::cout << "  Rotation drift before PGO: " << angleErr * 180.0 / CV_PI << " degrees" << std::endl;
                    
                    // Run pose graph optimization
                    optimizePoseGraph(poses, poseEdges, POSE_GRAPH_ITERATIONS);
                    
                    // Compute rotation drift after optimization
                    R_seq = poses[globalBestCurrFrame].R * poses[globalBestPastFrame].R.t();
                    R_err = globalBestR * R_seq.t();
                    cv::Rodrigues(R_err, rvec_err);
                    angleErr = cv::norm(rvec_err);
                    std::cout << "  Rotation drift after PGO: " << angleErr * 180.0 / CV_PI << " degrees" << std::endl;
                }
                
                // === ADD LOOP CLOSURE OBSERVATIONS ===
                int loopObsAdded = 0;
                for (size_t k = 0; k < globalBestMatches.size(); ++k)
                {
                    if (!globalBestMask.at<uchar>(static_cast<int>(k))) continue;
                    
                    int idxCurr = globalBestMatches[k].queryIdx;
                    int idxPast = globalBestMatches[k].trainIdx;
                    
                    int pointIdx = keypointToPointIdx[globalBestPastFrame][idxPast];
                    
                    if (pointIdx != -1)
                    {
                        // Add observation linking current frame to existing 3D point
                        observations.push_back({globalBestCurrFrame, pointIdx, 
                                               cv::Point2d(allKeypoints[globalBestCurrFrame][idxCurr].pt)});
                        keypointToPointIdx[globalBestCurrFrame][idxCurr] = pointIdx;
                        loopObsAdded++;
                    }
                }
                std::cout << "  Added " << loopObsAdded << " loop closure observations." << std::endl;
            }
            else
            {
                std::cout << "  No loop closure detected (gap=" << loopGap << " frames)." << std::endl;
            }
        }
        else
        {
            std::cout << "\n=== Loop Closure Detection DISABLED ===" << std::endl;
        }


        // --- 4. Report results ----------------------------------------------

        std::cout << "\n=== Reconstruction Summary ===" << std::endl;
        std::cout << "Number of keyframes: " << numViews << std::endl;
        std::cout << "Total 3D points: " << all3DPoints.size() << std::endl;
        std::cout << "Total observations: " << observations.size() << std::endl;
        
        // Print first and last few poses
        std::cout << "\nFirst keyframe pose (origin):" << std::endl;
        std::cout << "  R = I, t = [0,0,0]" << std::endl;
        
        if (numViews > 1) {
            std::cout << "\nLast keyframe pose (keyframe " << numViews-1 
                      << ", frame " << keyframeIndices[numViews-1] << "):" << std::endl;
            cv::Mat C = -poses[numViews-1].R.t() * poses[numViews-1].t;
            std::cout << "  Camera center: [" << C.at<double>(0) << ", " 
                      << C.at<double>(1) << ", " << C.at<double>(2) << "]" << std::endl;
        }

        // --- 5. Refine using interleaved bundle adjustment --------------------

        double errBefore = computeReprojectionError(K, poses, all3DPoints, observations);
        std::cout << "\nReprojection error BEFORE BA: " << errBefore << " px" << std::endl;
        
        runBundleAdjustment(K, poses, all3DPoints, observations, 5);

        double errAfter = computeReprojectionError(K, poses, all3DPoints, observations);
        std::cout << "\nReprojection error AFTER BA: " << errAfter << " px" << std::endl;
        
        // --- 6. Outlier Removal ----------------------------------------------
        // Remove points with high reprojection error or behind cameras
        
        std::cout << "\n=== Outlier Removal ===" << std::endl;
        
        // Compute per-point maximum reprojection error and check if behind any camera
        std::vector<bool> pointIsOutlier(all3DPoints.size(), false);
        std::vector<double> pointMaxError(all3DPoints.size(), 0.0);
        
        for (const auto& obs : observations)
        {
            const CameraPose& pose = poses[obs.camIndex];
            const cv::Point3d& X = all3DPoints[obs.pointIndex];
            
            // Check if behind camera
            cv::Mat Xw = (cv::Mat_<double>(3,1) << X.x, X.y, X.z);
            cv::Mat Xc = pose.R * Xw + pose.t;
            if (Xc.at<double>(2) <= 0) {
                pointIsOutlier[obs.pointIndex] = true;
                continue;
            }
            
            // Compute reprojection error
            double err = computeSingleReprojError(K, pose, X, obs.pixel);
            pointMaxError[obs.pointIndex] = std::max(pointMaxError[obs.pointIndex], err);
            
            if (err > OUTLIER_REPROJ_THRESHOLD) {
                pointIsOutlier[obs.pointIndex] = true;
            }
        }
        
        // Also mark points that are too far from the camera cluster
        // Compute centroid of camera centers
        cv::Mat centroid = cv::Mat::zeros(3, 1, CV_64F);
        int validCamCount = 0;
        for (const auto& pose : poses) {
            if (!pose.R.empty()) {
                cv::Mat C = -pose.R.t() * pose.t;
                centroid += C;
                validCamCount++;
            }
        }
        if (validCamCount > 0) centroid /= validCamCount;
        
        // Compute max camera distance from centroid
        double maxCamDist = 0;
        for (const auto& pose : poses) {
            if (!pose.R.empty()) {
                cv::Mat C = -pose.R.t() * pose.t;
                maxCamDist = std::max(maxCamDist, cv::norm(C - centroid));
            }
        }
        
        // Mark points too far from centroid (> 5x the camera spread)
        double distanceThreshold = std::max(10.0, maxCamDist * 5.0);
        for (size_t i = 0; i < all3DPoints.size(); ++i)
        {
            cv::Mat Xm = (cv::Mat_<double>(3,1) << all3DPoints[i].x, all3DPoints[i].y, all3DPoints[i].z);
            double dist = cv::norm(Xm - centroid);
            if (dist > distanceThreshold) {
                pointIsOutlier[i] = true;
            }
        }
        
        // Count outliers
        int nOutliers = 0;
        for (bool isOut : pointIsOutlier) {
            if (isOut) nOutliers++;
        }
        
        std::cout << "  Outliers detected: " << nOutliers << " / " << all3DPoints.size() 
                  << " (" << std::setprecision(1) << std::fixed 
                  << 100.0 * nOutliers / all3DPoints.size() << "%)" << std::endl;
        std::cout << "  Distance threshold: " << distanceThreshold << std::endl;
        
        // Create filtered point cloud and remap observations
        std::vector<cv::Point3d> filteredPoints;
        std::vector<int> oldToNewIdx(all3DPoints.size(), -1);
        
        for (size_t i = 0; i < all3DPoints.size(); ++i)
        {
            if (!pointIsOutlier[i]) {
                oldToNewIdx[i] = static_cast<int>(filteredPoints.size());
                filteredPoints.push_back(all3DPoints[i]);
            }
        }
        
        // Remap observations
        std::vector<Observation> filteredObs;
        for (const auto& obs : observations)
        {
            int newIdx = oldToNewIdx[obs.pointIndex];
            if (newIdx != -1) {
                filteredObs.push_back({obs.camIndex, newIdx, obs.pixel});
            }
        }
        
        std::cout << "  Points after filtering: " << filteredPoints.size() << std::endl;
        std::cout << "  Observations after filtering: " << filteredObs.size() << std::endl;
        
        // Replace with filtered data
        all3DPoints = std::move(filteredPoints);
        observations = std::move(filteredObs);
        
        // Run BA again on filtered data
        std::cout << "\n=== Final Bundle Adjustment ===" << std::endl;
        double errFiltered = computeReprojectionError(K, poses, all3DPoints, observations);
        std::cout << "Reprojection error after filtering: " << errFiltered << " px" << std::endl;
        
        runBundleAdjustment(K, poses, all3DPoints, observations, 3);
        
        double errFinal = computeReprojectionError(K, poses, all3DPoints, observations);
        std::cout << "\nFINAL reprojection error: " << errFinal << " px" << std::endl;
        
        // --- 7. Save to OBJ --------------------------------------------------

        // Save with timestamp
        std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::string file_name = "../data/reconstruction/reconstructionBundle_" + timestamp + ".obj";
        saveAsOBJ(file_name, all3DPoints, poses, SAVE_SCENE_POINTS);
        
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}