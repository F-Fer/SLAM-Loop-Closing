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
std::string VIDEO_FILENAME = "IMG_0280.MOV";

// Keyframe selection thresholds
const double MIN_MEDIAN_DISPLACEMENT = 20.0;   // Minimum median pixel displacement for keyframe
const double MAX_MEDIAN_DISPLACEMENT = 150.0;  // Maximum displacement (too much motion = blur)
const int MIN_TRACKED_FEATURES = 100;          // Minimum features to track
const double MIN_INLIER_RATIO = 0.3;           // Minimum inlier ratio for good pose
const int MIN_INLIERS_FOR_KEYFRAME = 50;       // Minimum inliers to accept keyframe


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
 * Uses simple Gauss-Newton with numeric Jacobian.
 *
 * @param K Intrinsic matrix.
 * @param camIndex Index of the camera to refine.
 * @param poses Vector of camera poses (will be updated at camIndex).
 * @param points3D All 3D points (fixed).
 * @param observations All observations.
 * @param maxIters Number of Gauss-Newton iterations.
 */
void refineCameraPoseGN(
    const cv::Mat& K,
    int camIndex,
    std::vector<CameraPose>& poses,
    const std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations,
    int maxIters = 10)
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
        // Build residual vector r
        const int m = static_cast<int>(obsForCam.size());
        cv::Mat r(2*m, 1, CV_64F);

        for (int i = 0; i < m; ++i)
        {
            const Observation* ob = obsForCam[i];
            const cv::Point3d& X = points3D[ob->pointIndex];

            CameraPose tmpPose;
            paramsToPose(p, tmpPose);
            cv::Point2d proj = projectPoint(K, tmpPose, X);

            r.at<double>(2*i+0) = proj.x - ob->pixel.x;
            r.at<double>(2*i+1) = proj.y - ob->pixel.y;
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
 * Uses simple Gauss-Newton with numeric Jacobian.
 *
 * @param K Intrinsic matrix.
 * @param pointIndex Index of the 3D point to refine.
 * @param poses All camera poses (fixed).
 * @param points3D Vector of 3D points (will be updated at pointIndex).
 * @param observations All observations.
 * @param maxIters Number of Gauss-Newton iterations.
 */
void refinePointGN(
    const cv::Mat& K,
    int pointIndex,
    const std::vector<CameraPose>& poses,
    std::vector<cv::Point3d>& points3D,
    const std::vector<Observation>& observations,
    int maxIters = 10)
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

        for (int i = 0; i < m; ++i)
        {
            const Observation* ob = obsForPoint[i];
            const CameraPose& pose = poses[ob->camIndex];

            cv::Point3d Xcur(p[0], p[1], p[2]);
            cv::Point2d proj = projectPoint(K, pose, Xcur);

            r.at<double>(2*i+0) = proj.x - ob->pixel.x;
            r.at<double>(2*i+1) = proj.y - ob->pixel.y;
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
    int numOuterIters = 5)
{
    if (poses.empty() || points3D.empty() || observations.empty())
    {
        std::cerr << "Nothing to optimize in BA." << std::endl;
        return;
    }

    std::cout << "\nStarting alternating bundle adjustment with "
              << numOuterIters << " outer iterations..." << std::endl;

    for (int iter = 0; iter < numOuterIters; ++iter)
    {
        std::cout << "  BA outer iteration " << iter+1 << " / "
                  << numOuterIters;

        // Pose-only step (Option C)
        for (int camIdx = 0; camIdx < static_cast<int>(poses.size()); ++camIdx)
        {
            refineCameraPoseGN(K, camIdx, poses, points3D, observations, 5);
        }

        // Point-only step (Option D)
        for (int ptIdx = 0; ptIdx < static_cast<int>(points3D.size()); ++ptIdx)
        {
            refinePointGN(K, ptIdx, poses, points3D, observations, 5);
        }

        double err = computeReprojectionError(K, poses, points3D, observations);
        std::cout << ": " << err << " px.\n";
    }

    std::cout << "Alternating bundle adjustment finished.\n";
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
    const std::vector<CameraPose> &poses)
{
    std::ofstream out(filename);
    if (!out.is_open())
    {
        std::cerr << "Could not write OBJ file: " << filename << std::endl;
        return;
    }

    out << "# OBJ file generated by OpenCV reconstruction demo\n";
    out << "# Number of 3D points: " << points3D.size() << "\n";
    out << "# Number of cameras: " << poses.size() << "\n\n";

    // --- Write 3D points -----------------------------------------------------
    out << "# 3D points\n";
    for (const auto &p : points3D)
    {
        out << "v " << p.x << " " << p.y << " " << p.z << "\n";
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
            
            // Estimate pose
            cv::Mat inlierMask, R, t;
            if (!estimateRelativePoseFromEssential(K, pts1, pts2, R, t, inlierMask))
            {
                continue;
            }
            
            int inlierCount = cv::countNonZero(inlierMask);
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
                      << std::endl;
            
            // Store keyframe data
            keyframeImages.push_back(imgUndistorted);
            allKeypoints.push_back(currKpts);
            allDescriptors.push_back(currDesc);
            keyframeIndices.push_back(static_cast<int>(frameIdx));
            keypointToPointIdx.push_back(std::vector<int>(currKpts.size(), -1));
            
            // Compute global pose
            CameraPose newPose;
            newPose.R = R * poses[lastKeyframeIdx].R;
            newPose.t = R * poses[lastKeyframeIdx].t + t;
            poses.push_back(newPose);
            
            // === TRIANGULATE AND MERGE MAP POINTS ===
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
            
            for (int k = 0; k < points4D.cols; ++k)
            {
                int matchIdx = inlierMatchIndices[k];
                int kptIdx1 = matches[matchIdx].queryIdx;  // Keypoint index in last keyframe
                int kptIdx2 = matches[matchIdx].trainIdx;  // Keypoint index in current frame
                
                cv::Mat col = points4D.col(k);
                double w = col.at<float>(3, 0);
                if (std::abs(w) < 1e-9) continue;
                
                double X = col.at<float>(0, 0) / w;
                double Y = col.at<float>(1, 0) / w;
                double Z = col.at<float>(2, 0) / w;
                
                cv::Point3d Xw(X, Y, Z);
                cv::Mat XwMat = (cv::Mat_<double>(3,1) << X, Y, Z);
                
                // Cheirality check
                cv::Mat Xc1 = poses[lastKeyframeIdx].R * XwMat + poses[lastKeyframeIdx].t;
                cv::Mat Xc2 = newPose.R * XwMat + newPose.t;
                if (Xc1.at<double>(2) <= 0 || Xc2.at<double>(2) <= 0) continue;
                
                // Distance filter
                if (cv::norm(XwMat) > 100.0) continue;
                
                // === MAP POINT MERGING ===
                // Check if the keypoint in the last keyframe already has an associated 3D point
                int existingPointIdx = keypointToPointIdx[lastKeyframeIdx][kptIdx1];
                
                if (existingPointIdx != -1)
                {
                    // MERGE: This feature was already triangulated from an earlier pair
                    // Just add an observation for the new keyframe to the existing point
                    observations.push_back({newKeyframeIdx, existingPointIdx, 
                                           cv::Point2d(inlierPts2[k].x, inlierPts2[k].y)});
                    keypointToPointIdx[newKeyframeIdx][kptIdx2] = existingPointIdx;
                    nMerged++;
                }
                else
                {
                    // NEW POINT: Create new 3D point
                    int pointIndex = static_cast<int>(all3DPoints.size());
                    all3DPoints.push_back(Xw);
                    
                    observations.push_back({lastKeyframeIdx, pointIndex, 
                                           cv::Point2d(inlierPts1[k].x, inlierPts1[k].y)});
                    observations.push_back({newKeyframeIdx, pointIndex, 
                                           cv::Point2d(inlierPts2[k].x, inlierPts2[k].y)});
                    
                    keypointToPointIdx[lastKeyframeIdx][kptIdx1] = pointIndex;
                    keypointToPointIdx[newKeyframeIdx][kptIdx2] = pointIndex;
                    nTriangulated++;
                }
            }
            
            std::cout << "  New points: " << nTriangulated << ", Merged: " << nMerged << std::endl;
            
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
            
            // === POSE GRAPH CORRECTION ===
            // Compute the expected relative pose from sequential tracking
            cv::Mat R_past = poses[globalBestPastFrame].R;
            cv::Mat t_past = poses[globalBestPastFrame].t;
            cv::Mat R_curr = poses[globalBestCurrFrame].R;
            cv::Mat t_curr = poses[globalBestCurrFrame].t;
            
            // Sequential relative pose: R_curr_from_past = R_curr * R_past^T
            cv::Mat R_seq = R_curr * R_past.t();
            cv::Mat t_seq = t_curr - R_seq * t_past;
            
            // Loop closure relative pose: globalBestR, globalBestT
            // This is the pose of curr relative to past from direct matching
            
            // Rotation error between sequential and loop closure
            cv::Mat R_err = globalBestR * R_seq.t();
            cv::Mat rvec_err;
            cv::Rodrigues(R_err, rvec_err);
            
            // Translation error (approximate - scale ambiguity exists)
            // We only correct rotation, as translation scale is ambiguous in monocular
            double angleErr = cv::norm(rvec_err);
            std::cout << "  Rotation drift: " << angleErr * 180.0 / CV_PI << " degrees" << std::endl;
            
            // Distribute rotation error linearly from past+1 to curr
            int numFramesToCorrect = globalBestCurrFrame - globalBestPastFrame;
            std::cout << "  Correcting poses for frames " << (globalBestPastFrame + 1) 
                      << " to " << globalBestCurrFrame << std::endl;
            
            for (int f = globalBestPastFrame + 1; f <= globalBestCurrFrame; ++f)
            {
                if (poses[f].R.empty()) continue;
                
                double alpha = (double)(f - globalBestPastFrame) / numFramesToCorrect;
                
                // Interpolate rotation correction using scaled axis-angle
                cv::Mat rvec_correction = alpha * rvec_err;
                cv::Mat R_correction;
                cv::Rodrigues(rvec_correction, R_correction);
                
                // Apply correction: R_new = R_correction * R_old
                poses[f].R = R_correction * poses[f].R;
                
                // Note: We don't correct translation due to scale ambiguity
                // The BA will handle small adjustments
            }
            
            std::cout << "  Pose graph correction applied." << std::endl;
            
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
        
        alternatingBundleAdjustment(
            K,
            poses,
            all3DPoints,
            observations,
            5   // number of outer iterations
        );

        double errAfter = computeReprojectionError(K, poses, all3DPoints, observations);
        std::cout << "\nReprojection error AFTER BA: " << errAfter << " px" << std::endl;
        
        // --- 6. Save to OBJ --------------------------------------------------

        // Save with timestamp
        std::string timestamp = std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
        std::string file_name = "../data/reconstruction/reconstructionBundle_" + timestamp + ".obj";
        saveAsOBJ(file_name, all3DPoints, poses);
        
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}
