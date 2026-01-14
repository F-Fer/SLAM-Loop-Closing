#include "extract_images.hpp"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <iomanip>

namespace fs = std::filesystem;

bool extract_images_from_mov(const std::string& video_path, const std::string& output_dir) {
    // Check if the video file exists
    if (!fs::exists(video_path)) {
        std::cerr << "Error: Video file does not exist: " << video_path << std::endl;
        return false;
    }

    // Create output directory if it doesn't exist
    if (!fs::exists(output_dir)) {
        if (!fs::create_directories(output_dir)) {
            std::cerr << "Error: Could not create directory: " << output_dir << std::endl;
            return false;
        }
    }

    // Open the video file
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file: " << video_path << std::endl;
        return false;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "Video opened successfully: " << video_path << std::endl;
    std::cout << "FPS: " << fps << ", Total Frames: " << total_frames << std::endl;

    cv::Mat frame;
    int frame_count = 0;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        // Generate filename: frame_0000.png, frame_0001.png, etc.
        std::stringstream ss;
        ss << output_dir << "/frame_" << std::setw(4) << std::setfill('0') << frame_count << ".png";
        std::string filename = ss.str();

        // Save the frame
        if (!cv::imwrite(filename, frame)) {
            std::cerr << "Error: Could not save frame " << frame_count << " to " << filename << std::endl;
            return false;
        }

        if (frame_count % 100 == 0) {
            std::cout << "Extracted frame " << frame_count << " / " << total_frames << std::endl;
        }
        frame_count++;
    }

    cap.release();
    std::cout << "Finished extraction. Total frames saved: " << frame_count << " to " << output_dir << std::endl;
    return true;
}
