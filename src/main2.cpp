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

int SKIP_FRAMES = 10;
std::string VIDEO_FILENAME = "IMG_0280.MOV";


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
    std::string video_path = "data/" + video_filename;
    if (!fs::exists(video_path)) {
        video_path = "../data/" + video_filename;
    }
    
    std::string video_name = video_filename.substr(0, video_filename.find('.'));
    std::string output_dir = "data/extracted_frames/" + video_name;
    if (!fs::exists("data") && fs::exists("../data")) {
        output_dir = "../data/extracted_frames/" + video_name;
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

        // find all frames and save the frame paths to a vector
        std::vector<std::string> argv;
        for (int i = 0; ; i += SKIP_FRAMES) {
            std::string frame_path = extracted_frames_dir + "/frame_" + std::format("{:04d}", i) + ".png";
            if (!fs::exists(frame_path)) break;
            
            argv.push_back(frame_path);
        }

        argc = argv.size();

        std::cout << "Loaded " << argc << " frames." << std::endl;

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
        // Distortion coefficients: [0.009593106889362086, -0.08836017837645339, -0.002369764239215277, -0.002095085353035259, 0.1736273482549004]
        cv::Mat distCoeffs = (cv::Mat_<double>(1, 5)
            << 0.009593106889362086, -0.08836017837645339,
            -0.002369764239215277, -0.002095085353035259,
            0.1736273482549004);

        

        // --- 2. Load input images -------------------------------------------

        std::vector<cv::Mat> images;
        images.reserve(argc - 1);

        for (int i = 0; i < argc; ++i)
        {
            cv::Mat img = cv::imread(argv[i], cv::IMREAD_GRAYSCALE);
            if (img.empty())
            {
                std::cerr << "Failed to load image: " << argv[i] << std::endl;
                return EXIT_FAILURE;
            }
            images.push_back(img);
        }

        const int numViews = static_cast<int>(images.size());
        std::cout << "Loaded " << numViews << " images." << std::endl;

        // --- 3. Detect SIFT features & descriptors for each image -----------

        std::vector<std::vector<cv::KeyPoint>> allKeypoints(numViews);
        std::vector<cv::Mat> allDescriptors(numViews);

        for (int i = 0; i < numViews; ++i)
        {
            detectAndDescribeSIFT(images[i], allKeypoints[i], allDescriptors[i]);
            std::cout << "Image " << i << ": " << allKeypoints[i].size()
                      << " keypoints." << std::endl;
        }

        // --- 4. Initialize camera poses -------------------------------------
        //
        // We choose the first camera as the world reference:
        // R_0 = Identity, t_0 = 0. Subsequent poses are relative to this.
        std::vector<CameraPose> poses(numViews);

        poses[0].R = cv::Mat::eye(3, 3, CV_64F);
        poses[0].t = cv::Mat::zeros(3, 1, CV_64F);

        // --- 5. For each consecutive pair, estimate relative pose and triangulate

        std::vector<cv::Point3d> all3DPoints; // aggregated 3D points from all pairs
        std::vector<Observation> observations;
        
        for (int i = 0; i < numViews - 1; ++i)
        {
            std::cout << "\nProcessing image pair " << i << " -> " << (i + 1) << " ..." << std::endl;

            // 5.1 Match features between view i and i+1
            std::vector<cv::DMatch> matches;
            matchFeatures(allDescriptors[i], allDescriptors[i+1], matches);

            std::cout << "  Raw good matches after ratio test: " << matches.size() << std::endl;
            if (matches.size() < 20)
            {
                std::cerr << "  Too few matches between images " << i << " and " << (i + 1)
                          << ". Skipping this pair." << std::endl;
                continue;
            }

            std::vector<cv::Point2f> pts1, pts2;
            extractMatchedPoints(allKeypoints[i], allKeypoints[i+1], matches, pts1, pts2);

            // 5.2 Estimate essential matrix and relative pose
            cv::Mat inlierMask;
            cv::Mat R, t;

            if (!estimateRelativePoseFromEssential(K, pts1, pts2, R, t, inlierMask))
            {
                std::cerr << "  Pose estimation failed for pair " << i << " -> " << (i + 1) << std::endl;
                continue;
            }

            int inlierCount = cv::countNonZero(inlierMask);
            std::cout << "  Inliers after essential matrix + recoverPose: "
                      << inlierCount << std::endl;

            // 5.3 Compute global pose for camera (i+1)
            //
            // recoverPose gives transform from camera i to camera (i+1):
            // X_{i+1} = R * X_i + t
            //
            // If pose[i] transforms world -> camera i:
            //   X_i = R_i * X_w + t_i
            //
            // Then:
            //   X_{i+1} = R * (R_i X_w + t_i) + t = (R R_i) X_w + (R t_i + t)
            //
            // => R_{i+1} = R * R_i, t_{i+1} = R * t_i + t

            // If previous pose was not computed (e.g. tracking failure), reset to identity
            if (poses[i].R.empty()) {
                std::cout << "  [WARNING] Previous pose (camera " << i << ") invalid. Resetting to Identity." << std::endl;
                poses[i].R = cv::Mat::eye(3, 3, CV_64F);
                poses[i].t = cv::Mat::zeros(3, 1, CV_64F);
            }

            poses[i+1].R = R * poses[i].R;
            poses[i+1].t = R * poses[i].t + t;

            // 5.4 Triangulate 3D points for this inlier set
            std::vector<cv::Point2f> inlierPts1, inlierPts2;
            inlierPts1.reserve(inlierCount);
            inlierPts2.reserve(inlierCount);

            for (size_t k = 0; k < pts1.size(); ++k)
            {
                if (inlierMask.at<uchar>(static_cast<int>(k)))
                {
                    inlierPts1.push_back(pts1[k]);
                    inlierPts2.push_back(pts2[k]);
                }
            }

            // Projection matrices P1 and P2 (world coordinates -> image pixels)
            cv::Mat P1(3, 4, CV_64F);
            cv::Mat P2(3, 4, CV_64F);

            {
                // P = K [R | t]
                cv::hconcat(poses[i].R, poses[i].t, P1);
                P1 = K * P1;

                cv::hconcat(poses[i+1].R, poses[i+1].t, P2);
                P2 = K * P2;
            }

            cv::Mat points4D;
            cv::triangulatePoints(P1, P2, inlierPts1, inlierPts2, points4D);

            int nTriangulated = 0;

            for (int k = 0; k < points4D.cols; ++k)
            {
                cv::Mat col = points4D.col(k);
                double w = col.at<float>(3, 0);
                if (std::abs(w) < 1e-9)
                    continue;

                double X = col.at<float>(0, 0) / w;
                double Y = col.at<float>(1, 0) / w;
                double Z = col.at<float>(2, 0) / w;

                cv::Point3d Xw(X, Y, Z);

                // Simple cheirality check: point in front of both cameras
                cv::Mat XwMat = (cv::Mat_<double>(3,1) << X, Y, Z);
                cv::Mat Xc1 = poses[i].R * XwMat + poses[i].t;
                cv::Mat Xc2 = poses[i+1].R * XwMat + poses[i+1].t;

                if (Xc1.at<double>(2) <= 0 || Xc2.at<double>(2) <= 0)
                    continue;

                int pointIndex = static_cast<int>(all3DPoints.size());
                all3DPoints.push_back(Xw);
                nTriangulated++;

                // Observations: one in each image
                observations.push_back({i,   pointIndex,
                                        cv::Point2d(inlierPts1[k].x, inlierPts1[k].y)});
                observations.push_back({i+1, pointIndex,
                                        cv::Point2d(inlierPts2[k].x, inlierPts2[k].y)});
            }
        }

        // --- 6. Report results ----------------------------------------------

        std::cout << "\n=== Reconstruction Summary ===" << std::endl;
        std::cout << "Number of views: " << numViews << std::endl;
        std::cout << "Total 3D points (unfiltered, from all pairs): "
                  << all3DPoints.size() << std::endl;

        for (int i = 0; i < numViews; ++i)
        {
            std::cout << "\nCamera " << i << " pose (R|t):" << std::endl;
            std::cout << "R = " << std::endl << poses[i].R << std::endl;
            std::cout << "t = " << std::endl << poses[i].t << std::endl;
        }

        // --- 7. Refine using interleaved bundle adjustment --------------------

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
        
        // --- 8. Save to OBJ --------------------------------------------------

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
