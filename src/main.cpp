#include "extract_images.hpp"
#include "loop_closing.hpp"
#include <iostream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int extract_images() {
    std::string video_path = "data/IMG_0242.MOV";
    if (!fs::exists(video_path)) {
        video_path = "../data/IMG_0242.MOV";
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

int run_loop_closing() {
    std::cout << "\n=== Starting Loop Closing System ===" << std::endl;
    
    // Find frames directory
    std::string frames_dir = "data/extracted_frames";
    if (!fs::exists(frames_dir) && fs::exists("../data/extracted_frames")) {
        frames_dir = "../data/extracted_frames";
    }
    
    if (!fs::exists(frames_dir)) {
        std::cerr << "Error: Frames directory not found: " << frames_dir << std::endl;
        std::cerr << "Please run image extraction first." << std::endl;
        return -1;
    }
    
    // Get all frame files
    std::vector<std::string> frame_files;
    for (const auto& entry : fs::directory_iterator(frames_dir)) {
        if (entry.path().extension() == ".png") {
            frame_files.push_back(entry.path().string());
        }
    }
    
    // Sort frame files
    std::sort(frame_files.begin(), frame_files.end());
    
    std::cout << "Found " << frame_files.size() << " frames" << std::endl;
    
    if (frame_files.empty()) {
        std::cerr << "Error: No frames found in " << frames_dir << std::endl;
        return -1;
    }
    
    // Initialize loop closing system
    // Parameters: loop_threshold (0.0-1.0), min_loop_gap (frames)
    loop_closing::LoopClosingSystem system(0.15, 30);
    
    // Process every Nth frame to speed up processing (adjust as needed)
    int frame_skip = 3;  // Process every 3rd frame
    
    std::cout << "Processing frames (every " << frame_skip << " frames)..." << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    
    for (size_t i = 0; i < frame_files.size(); i += frame_skip) {
        cv::Mat image = cv::imread(frame_files[i]);
        if (image.empty()) {
            std::cerr << "Warning: Could not read frame " << frame_files[i] << std::endl;
            continue;
        }
        
        // Resize for faster processing (optional)
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(), 0.5, 0.5);
        
        system.processFrame(resized, i);
        
        // Show progress
        if (i % 30 == 0) {
            std::cout << "Processed frame " << i << "/" << frame_files.size() << std::endl;
        }
    }
    
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "\n=== Processing Complete ===" << std::endl;
    std::cout << "Total frames processed: " << system.getFrames().size() << std::endl;
    std::cout << "Loop closures detected: " << system.getLoopClosures().size() << std::endl;
    
    // Save results
    std::string output_dir = "data/loop_closing_results";
    if (!fs::exists("data") && fs::exists("../data")) {
        output_dir = "../data/loop_closing_results";
    }
    
    system.saveResults(output_dir);
    
    std::cout << "\n=== Loop Closing System Complete ===" << std::endl;
    
    return 0;
}

int main(int argc, char** argv) {
    std::cout << "SLAM Loop Closing System" << std::endl;
    std::cout << "========================" << std::endl << std::endl;
    
    // Check command line arguments
    if (argc > 1) {
        std::string mode = argv[1];
        if (mode == "extract") {
            return extract_images();
        } else if (mode == "loop") {
            return run_loop_closing();
        } else if (mode == "all") {
            int ret = extract_images();
            if (ret != 0) return ret;
            return run_loop_closing();
        } else {
            std::cout << "Usage: " << argv[0] << " [extract|loop|all]" << std::endl;
            std::cout << "  extract - Extract frames from video" << std::endl;
            std::cout << "  loop    - Run loop closing on extracted frames" << std::endl;
            std::cout << "  all     - Run both extraction and loop closing" << std::endl;
            return 0;
        }
    }
    
    // Default: run loop closing (assuming frames are already extracted)
    return run_loop_closing();
}
