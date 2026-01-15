#include "extract_images.hpp"
#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <format>
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

int loop_closing() {
    std::string video_path = "data/IMG_0243.MOV";
    if (!fs::exists(video_path)) {
        video_path = "../data/IMG_0243.MOV";
    }
    
    std::string output_dir = "data/loop_closing_results";
    if (!fs::exists("data") && fs::exists("../data")) {
        output_dir = "../data/loop_closing_results";
    }

    // Get the total number of frames
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

    // Run first frame
    std::string first_frame_path = extracted_frames_dir + "/frame_" + std::format("{:04d}", current_frame_index) + ".png";
    current_frame = cv::imread(first_frame_path);
    if (current_frame.empty()) {
        std::cerr << "Could not read first frame: " << first_frame_path << std::endl;
        return -1;
    }
    orb->detectAndCompute(current_frame, cv::noArray(), current_features, current_descriptors);
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

        // Match features
        matcher.match(previous_descriptors, current_descriptors, matches);
        
        std::cout << "Frame " << current_frame_index << " - Matches: " << matches.size() << std::endl;

        // Triangulation
        // 1. Convert KeyPoints to Points
        std::vector<cv::Point2f> pts1, pts2;
        for (const auto& m : matches) {
            pts1.push_back(previous_features[m.queryIdx].pt);
            pts2.push_back(current_features[m.trainIdx].pt);
        }

        // 2. Find essential matrix with correct intrinsics
        double focal = 712.8;
        cv::Point2d pp(540, 960); // assuming the principal point is at the center of the image
        cv::Mat mask;
        cv::Mat essential_matrix = cv::findEssentialMat(pts1, pts2, focal, pp, cv::RANSAC, 0.999, 1.0, mask);

        // 3. Recover pose
        cv::Mat R, t;
        cv::recoverPose(essential_matrix, pts1, pts2, R, t, focal, pp, mask);

        

        current_frame_index++;
    }

    return 0;
}

int main() {
    // extract_images();
    loop_closing();
    return 0;
}
