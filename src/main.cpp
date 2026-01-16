#include "extract_images.hpp"
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <format>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>


namespace fs = std::filesystem;

int extract_images() {
    std::string video_path = "data/IMG_0243.MOV";
    if (!fs::exists(video_path)) {
        video_path = "../data/IMG_0243.MOV";
    }
    
    std::string output_dir = "data/extracted_frames";
    if (!fs::exists("data") && fs::exists("../data")) {
        output_dir = "../data/extracted_frames";
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

int readFrame(const std::string& frame_path, cv::Mat& frame) {
    frame = cv::imread(frame_path);
    if (frame.empty()) {
        std::cerr << "Could not read frame: " << frame_path << std::endl;
        return -1;
    }
    return 0;
}

void saveTrajectoryToPLY(const std::vector<cv::Mat>& poses, const std::string& filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Could not open file for writing: " << filename << std::endl;
        return;
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << poses.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (size_t i = 0; i < poses.size(); ++i) {
        cv::Mat R = poses[i](cv::Rect(0, 0, 3, 3));
        cv::Mat t = poses[i](cv::Rect(3, 0, 1, 3));
        
        // Camera center in world coordinates: C = -R^T * t
        cv::Mat C = -R.t() * t;
        
        // Color gradient from red (start) to blue (end)
        int r = static_cast<int>(255 * (1.0 - static_cast<double>(i) / std::max(1UL, poses.size() - 1)));
        int b = static_cast<int>(255 * (static_cast<double>(i) / std::max(1UL, poses.size() - 1)));
        int g = 0;

        out << C.at<double>(0) << " " << C.at<double>(1) << " " << C.at<double>(2) << " "
            << r << " " << g << " " << b << "\n";
    }
    out.close();
    std::cout << "Trajectory saved to " << filename << std::endl;
}

int loop_closing() {
    std::string video_path = "data/IMG_0243.MOV";
    if (!fs::exists(video_path)) {
        video_path = "../data/IMG_0243.MOV";
    }
    
    std::string output_dir = "data/loop_closing_results";
    if (!fs::exists("data") && fs::exists("../data")) {
        output_dir = "../data/loop_closing_results";
    }

    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }

    // Find the extracted frames directory
    std::string extracted_frames_dir = "data/extracted_frames";
    if (!fs::exists(extracted_frames_dir)) {
        extracted_frames_dir = "../data/extracted_frames";
        if (!fs::exists(extracted_frames_dir)) {
            std::cerr << "Could not find extracted frames directory: " << extracted_frames_dir << std::endl;
            return -1;
        }
    }

    // Initialize variables for loop closing
    bool found_new_frames = true;
    int current_frame_index = 0;
    cv::Mat current_frame;
    cv::Mat previous_frame;
    std::vector<cv::KeyPoint> current_features;
    cv::Mat current_descriptors;
    std::vector<cv::KeyPoint> previous_features;
    cv::Mat previous_descriptors;
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    cv::BFMatcher matcher(cv::NORM_HAMMING); // Use Hamming distance
    std::vector<cv::DMatch> matches;

    // Track trajectory
    cv::Mat T_total = cv::Mat::eye(4, 4, CV_64F);
    std::vector<cv::Mat> all_poses;
    all_poses.push_back(T_total.clone());

    // --------------------------------------------------------------------------------------------------
    // Initialization (frame 0 & 1)

    // Camera matrix
    double focal = 712.8;
    cv::Point2d pp(540, 960); // assuming the principal point is at the center of the image
    cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

    // Get first two frames
    std::string first_frame_path = extracted_frames_dir + "/frame_" + std::format("{:04d}", current_frame_index) + ".png";
    readFrame(first_frame_path, previous_frame);
    current_frame_index++;
    std::string second_frame_path = extracted_frames_dir + "/frame_" + std::format("{:04d}", current_frame_index) + ".png";
    readFrame(second_frame_path, current_frame);

    // Detect features
    orb->detectAndCompute(previous_frame, cv::noArray(), previous_features, previous_descriptors);
    orb->detectAndCompute(current_frame, cv::noArray(), current_features, current_descriptors);

    // Match features
    matcher.match(previous_descriptors, current_descriptors, matches);
    std::cout << "Matches between frame 0 and 1: " << matches.size() << std::endl;

    // Filter matches (simple distance-based filter)
    std::sort(matches.begin(), matches.end());
    if (matches.size() > 500) matches.erase(matches.begin() + 500, matches.end());

    // 1. Convert KeyPoints to Points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : matches) {
        pts1.push_back(previous_features[m.queryIdx].pt);
        pts2.push_back(current_features[m.trainIdx].pt);
    }

    // 2. Find essential matrix
    cv::Mat mask;
    cv::Mat essential_matrix = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

    // 3. Recover pose
    cv::Mat R, t;
    cv::recoverPose(essential_matrix, pts1, pts2, K, R, t, mask);

    cv::Mat R_t = R.t();

    // 4. Triangulate
    // Projection matrices
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    cv::Mat Rt;
    cv::hconcat(R_t, -R_t * t, Rt);
    cv::Mat P2 = K * Rt;

    cv::Mat points4D;
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

    // Convert to 3D points and store them
    std::vector<cv::Point3f> world_points;
    std::vector<cv::Point2f> tracked_2d_points;
    cv::Mat tracked_descriptors;

    for (int i = 0; i < points4D.cols; i++) {
        float w = points4D.at<float>(3, i);
        if (std::abs(w) > 1e-6) {
            cv::Point3f p(points4D.at<float>(0, i) / w,
                          points4D.at<float>(1, i) / w,
                          points4D.at<float>(2, i) / w);
            
            // Check if point is in front of both cameras
            if (p.z > 0) {
                // Transform to second camera frame to check depth there too
                cv::Mat p_cam2 = R_t * (cv::Mat_<double>(3,1) << p.x, p.y, p.z) - R_t * t;
                if (p_cam2.at<double>(2) > 0) {
                    world_points.push_back(p);
                    tracked_2d_points.push_back(pts2[i]);
                    tracked_descriptors.push_back(current_descriptors.row(matches[i].trainIdx));
                }
            }
        }
    }

    std::cout << "Initial triangulation: " << world_points.size() << " points" << std::endl;

    // Update T_total for frame 1
    // T_total currently is Frame 0 (Identity)
    // T_1_w = [R.t() | -R.t()*t]
    cv::Mat T1 = cv::Mat::eye(4, 4, CV_64F);
    R_t.copyTo(T1(cv::Rect(0, 0, 3, 3)));
    cv::Mat t_world = -R_t * t;
    t_world.copyTo(T1(cv::Rect(3, 0, 1, 3)));
    
    T_total = T1;
    all_poses.push_back(T_total.clone());

    current_frame_index++;

    // Loop over remaining frames
    while (true) {
        // Move to next frame
        previous_frame = current_frame.clone();
        previous_features = current_features;
        previous_descriptors = current_descriptors.clone();

        // Read current frame
        std::string current_frame_path = extracted_frames_dir + "/frame_" + std::format("{:04d}", current_frame_index) + ".png"; // Make sure to use 4 digits for the frame index
        current_frame = cv::imread(current_frame_path);
        if (current_frame.empty()) {
            break;
        }

        // Detect and compute features
        orb->detectAndCompute(current_frame, cv::noArray(), current_features, current_descriptors);

        // --- STEP 3: TRACKING (Map-to-Frame) ---
        // Match descriptors of existing 3D points against current frame
        std::vector<cv::DMatch> map_matches;
        matcher.match(tracked_descriptors, current_descriptors, map_matches);
        
        std::vector<cv::Point3f> pnp_world_points;
        std::vector<cv::Point2f> pnp_image_points;
        for (const auto& m : map_matches) {
            pnp_world_points.push_back(world_points[m.queryIdx]);
            pnp_image_points.push_back(current_features[m.trainIdx].pt);
        }

        if (pnp_world_points.size() < 10) {
            std::cerr << "Tracking lost at frame " << current_frame_index << std::endl;
            break;
        }

        // Recover current pose [R|t] relative to world
        cv::Mat rvec, t_pnp;
        cv::solvePnPRansac(pnp_world_points, pnp_image_points, K, cv::Mat(), rvec, t_pnp);
        cv::Mat R_curr;
        cv::Rodrigues(rvec, R_curr);

        // Update T_total (Camera-to-World)
        cv::Mat T_curr = cv::Mat::eye(4, 4, CV_64F);
        cv::Mat R_w = R_curr.t();
        cv::Mat t_w = -R_curr.t() * t_pnp;
        R_w.copyTo(T_curr(cv::Rect(0, 0, 3, 3)));
        t_w.copyTo(T_curr(cv::Rect(3, 0, 1, 3)));
        T_total = T_curr;
        all_poses.push_back(T_total.clone());

        // --- STEP 4: TRIANGULATION (Map Extension) ---
        // 1. Match previous frame to current frame to find potential new points
        std::vector<cv::DMatch> frame_matches;
        matcher.match(previous_descriptors, current_descriptors, frame_matches);

        // 2. Get projection matrices
        // P = K * [R_world_to_cam | t_world_to_cam]
        cv::Mat T_prev_inv = all_poses[all_poses.size() - 2].inv();
        cv::Mat P_prev = K * T_prev_inv(cv::Rect(0, 0, 4, 3));
        
        cv::Mat T_curr_inv = T_total.inv();
        cv::Mat P_curr = K * T_curr_inv(cv::Rect(0, 0, 4, 3));

        std::vector<cv::Point2f> pts_prev, pts_curr;
        for (const auto& m : frame_matches) {
            pts_prev.push_back(previous_features[m.queryIdx].pt);
            pts_curr.push_back(current_features[m.trainIdx].pt);
        }

        cv::Mat points4D;
        cv::triangulatePoints(P_prev, P_curr, pts_prev, pts_curr, points4D);

        // 3. Add new valid points to the map
        tracked_descriptors.release(); // We will rebuild this
        std::vector<cv::Point3f> new_world_points;
        cv::Mat new_tracked_descriptors;

        for (int i = 0; i < points4D.cols; i++) {
            float w = points4D.at<float>(3, i);
            if (std::abs(w) > 1e-6) {
                cv::Point3f p(points4D.at<float>(0, i)/w, points4D.at<float>(1, i)/w, points4D.at<float>(2, i)/w);
                
                // Basic filtering: point must be in front of camera and not too far
                cv::Mat p_local = R_curr * (cv::Mat_<double>(3,1) << p.x, p.y, p.z) + t_pnp;
                if (p_local.at<double>(2) > 0 && p_local.at<double>(2) < 50.0) {
                    new_world_points.push_back(p);
                    new_tracked_descriptors.push_back(current_descriptors.row(frame_matches[i].trainIdx));
                }
            }
        }
        
        // Update the map for the next frame
        world_points = new_world_points;
        tracked_descriptors = new_tracked_descriptors;

        std::cout << "Frame " << current_frame_index << " - PnP Inliers: " << pnp_world_points.size() 
                    << " - Map Size: " << world_points.size() << std::endl;

        current_frame_index++;
    }

    // Save the trajectory
    saveTrajectoryToPLY(all_poses, output_dir + "/trajectory.ply");

    return 0;
}

int main() {
    // extract_images();
    loop_closing();
    return 0;
}
