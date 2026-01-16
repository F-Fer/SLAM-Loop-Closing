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

int SKIP_FRAMES = 3;

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

    // Calculate all camera centers first to find the centroid
    std::vector<cv::Mat> centers;
    cv::Mat mean_center = cv::Mat::zeros(3, 1, CV_64F);
    for (const auto& pose : poses) {
        // The poses are stored as Camera-to-World matrices (T_wc).
        // The translation part of T_wc represents the camera center in world coordinates.
        cv::Mat t = pose(cv::Rect(3, 0, 1, 3));
        // cv::Mat C = -R.t() * t;
        // centers.push_back(C.clone());
        // mean_center += C;
        centers.push_back(t.clone());
        mean_center += t;
    }
    if (!poses.empty()) {
        mean_center /= static_cast<double>(poses.size());
    }

    for (size_t i = 0; i < centers.size(); ++i) {
        // Subtract mean to center at 0,0,0
        cv::Mat C_centered = centers[i] - mean_center;
        
        // Color gradient from red (start) to blue (end)
        int r = static_cast<int>(255 * (1.0 - static_cast<double>(i) / std::max(1UL, poses.size() - 1)));
        int b = static_cast<int>(255 * (static_cast<double>(i) / std::max(1UL, poses.size() - 1)));
        int g = 0;

        out << C_centered.at<double>(0) << " " << C_centered.at<double>(1) << " " << C_centered.at<double>(2) << " "
            << r << " " << g << " " << b << "\n";
    }
    out.close();
    std::cout << "Trajectory centered and saved to " << filename << std::endl;
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

    // --------------------------------------------------------------------------------------------------
    // 1. heavy lifting: extract all frames and their features

    // Extract all frames into a vector
    std::vector<cv::Mat> frames;
    int total_processed_frames = 0;
    for (int i = 0; ; i += SKIP_FRAMES) {
        std::string frame_path = extracted_frames_dir + "/frame_" + std::format("{:04d}", i) + ".png";
        if (!fs::exists(frame_path)) break;
        
        cv::Mat frame;
        if (readFrame(frame_path, frame) == 0) {
            frames.push_back(frame);
        }
        total_processed_frames = i + SKIP_FRAMES;
    }

    // Extract features from all frames
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<std::vector<cv::KeyPoint>> frame_keypoints; // [frame_index][keypoint_index]
    std::vector<cv::Mat> frame_descriptors; // [frame_index][descriptor_index]
    for (const auto& frame : frames) {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(frame, cv::noArray(), keypoints, descriptors);
        frame_keypoints.push_back(keypoints);
        frame_descriptors.push_back(descriptors);
    }

    std::cout << "Processed up to frame " << total_processed_frames << ", " << frames.size() << " frames were loaded" << std::endl;

    // Initialize variables for loop closing
    cv::Mat T_total = cv::Mat::eye(4, 4, CV_64F);
    std::vector<cv::Mat> all_poses;
    all_poses.push_back(T_total.clone());

    // --------------------------------------------------------------------------------------------------
    // Initialization (frame 0 & 1)

    // Camera matrix
    double focal = 712.8;
    cv::Point2d pp(540, 960); // assuming the principal point is at the center of the image
    cv::Mat K = (cv::Mat_<double>(3, 3) << focal, 0, pp.x, 0, focal, pp.y, 0, 0, 1);

    // Match features between frame 0 and 1
    cv::BFMatcher matcher(cv::NORM_HAMMING); // Use Hamming distance
    std::vector<cv::DMatch> init_matches;
    matcher.match(frame_descriptors[0], frame_descriptors[1], init_matches);
    std::cout << "Matches between frame 0 and 1: " << init_matches.size() << std::endl;

    // Filter matches (simple distance-based filter)
    std::sort(init_matches.begin(), init_matches.end()); // sort matches by distance
    if (init_matches.size() > 500) init_matches.erase(init_matches.begin() + 500, init_matches.end()); // keep only the best 500 matches

    // convert KeyPoints to Points
    std::vector<cv::Point2f> pts1, pts2;
    for (const auto& m : init_matches) {
        pts1.push_back(frame_keypoints[0][m.queryIdx].pt);
        pts2.push_back(frame_keypoints[1][m.trainIdx].pt);
    }

    // find essential matrix
    cv::Mat mask;
    cv::Mat essential_matrix = cv::findEssentialMat(pts1, pts2, K, cv::RANSAC, 0.999, 1.0, mask);

    // recover pose of camera 1 relative to camera 0
    cv::Mat R, t;
    cv::recoverPose(essential_matrix, pts1, pts2, K, R, t, mask);

    // build projection matrices
    // P1 = K [I|0]
    cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F); 
    // P2 = K [R|t] (World-to-Camera for frame 1)
    // This assumes Camera 1 is the world origin, so P1 = K[I|0].
    // recoverPose returns R, t from Camera 0 to Camera 1 (pts1 to pts2).
    // So [R|t] is T_c1_c0. If C0 is world, this is T_c1_w.
    cv::Mat Rt_init;
    cv::hconcat(R, t, Rt_init);
    cv::Mat P2 = K * Rt_init;

    // test old triangulation method ---
    // cv::Mat R_t = R.t();
    // cv::Mat P1 = K * cv::Mat::eye(3, 4, CV_64F);
    // cv::Mat Rt;
    // cv::hconcat(R_t, -R_t * t, Rt);
    // cv::Mat P2 = K * Rt;
    // ---

    cv::Mat points4D; 
    cv::triangulatePoints(P1, P2, pts1, pts2, points4D);

    // convert 4D points to 3D points and store them
    std::vector<cv::Point3f> world_points; 
    cv::Mat tracked_descriptors; 

    for (int j = 0; j < points4D.cols; j++) {
        float w = points4D.at<float>(3, j);
        if (std::abs(w) > 1e-6) {
            cv::Point3f p(points4D.at<float>(0, j) / w,
                          points4D.at<float>(1, j) / w,
                          points4D.at<float>(2, j) / w);
            
            // Check if point is in front of both cameras
            if (p.z > 0) {
                // Transform to second camera frame to check depth there too: x2 = R*p + t
                cv::Mat p_cam2 = R * (cv::Mat_<double>(3,1) << p.x, p.y, p.z) + t;
                // cv::Mat p_cam2 = R_t * (cv::Mat_<double>(3,1) << p.x, p.y, p.z) - R_t * t; // test old triangulation method
                if (p_cam2.at<double>(2) > 0) {
                    world_points.push_back(p);
                    tracked_descriptors.push_back(frame_descriptors[1].row(init_matches[j].trainIdx));
                }
            }
        }
    }

    std::cout << "Initial triangulation: " << world_points.size() << " points" << std::endl;

    // Update T_total for frame 1 (Camera-to-World)
    // T_1_w = T_w_1.inv() = [R|t].inv() = [R^T | -R^T * t]
    cv::Mat T1 = cv::Mat::eye(4, 4, CV_64F);
    cv::Mat R_w_init = R.t();
    cv::Mat t_w_init = -R.t() * t;
    R_w_init.copyTo(T1(cv::Rect(0, 0, 3, 3)));
    t_w_init.copyTo(T1(cv::Rect(3, 0, 1, 3)));

    // old triangulation method ---
    // R_t.copyTo(T1(cv::Rect(0, 0, 3, 3)));
    // cv::Mat t_world = -R_t * t;
    // t_world.copyTo(T1(cv::Rect(3, 0, 1, 3)));
    // ---
    
    T_total = T1;
    all_poses.push_back(T_total.clone());

    // Loop over remaining frames
    for (int i = 2; i < frames.size(); i++) {
        // Match descriptors of existing 3D points against current frame
        std::vector<cv::DMatch> map_matches;
        matcher.match(tracked_descriptors, frame_descriptors[i], map_matches);
        
        std::vector<cv::Point3f> pnp_world_points; 
        std::vector<cv::Point2f> pnp_image_points; 
        for (const auto& m : map_matches) {
            pnp_world_points.push_back(world_points[m.queryIdx]); 
            pnp_image_points.push_back(frame_keypoints[i][m.trainIdx].pt); 
        }

        if (pnp_world_points.size() < 10) {
            std::cerr << "Tracking lost at frame index " << i << std::endl;
            break;
        }

        // Recover current pose [R|t] relative to world
        cv::Mat rvec, t_pnp, inliers;
        cv::solvePnPRansac(pnp_world_points, pnp_image_points, K, cv::Mat(), rvec, t_pnp, false, 100, 8.0, 0.99, inliers);
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

        // match previous frame to current frame to find potential new points
        std::vector<cv::DMatch> frame_matches;
        matcher.match(frame_descriptors[i-1], frame_descriptors[i], frame_matches);

        // get projection matrices (World-to-Camera)
        cv::Mat T_prev_inv = all_poses[all_poses.size() - 2].inv(); 
        cv::Mat P_prev = K * T_prev_inv(cv::Rect(0, 0, 4, 3)); 
        
        cv::Mat T_curr_inv = T_total.inv(); 
        cv::Mat P_curr = K * T_curr_inv(cv::Rect(0, 0, 4, 3)); 

        std::vector<cv::Point2f> pts_prev, pts_curr; 
        for (const auto& m : frame_matches) {
            pts_prev.push_back(frame_keypoints[i-1][m.queryIdx].pt); 
            pts_curr.push_back(frame_keypoints[i][m.trainIdx].pt); 
        }

        cv::Mat loop_points4D; 
        cv::triangulatePoints(P_prev, P_curr, pts_prev, pts_curr, loop_points4D);

        // add new valid points to the map
        std::vector<cv::Point3f> new_world_points; 
        cv::Mat new_tracked_descriptors; 

        for (int j = 0; j < loop_points4D.cols; j++) { 
            float w = loop_points4D.at<float>(3, j);
            if (std::abs(w) > 1e-6) {
                cv::Point3f p(loop_points4D.at<float>(0, j)/w, loop_points4D.at<float>(1, j)/w, loop_points4D.at<float>(2, j)/w);
                
                // transform to current camera frame to check depth
                cv::Mat p_local = R_curr * (cv::Mat_<double>(3,1) << p.x, p.y, p.z) + t_pnp;
                if (p_local.at<double>(2) > 0 && p_local.at<double>(2) < 50.0) {
                    new_world_points.push_back(p);
                    new_tracked_descriptors.push_back(frame_descriptors[i].row(frame_matches[j].trainIdx));
                }
            }
        }
        
        // update the map for the next frame
        world_points = new_world_points;
        tracked_descriptors = new_tracked_descriptors;

        std::cout << "Frame " << i * SKIP_FRAMES << " - PnP Inliers: " << inliers.rows 
                    << " - Map Size: " << world_points.size() << std::endl;
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
