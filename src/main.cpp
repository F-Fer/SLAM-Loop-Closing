#include "extract_images.hpp"
#include <iostream>
#include <filesystem>

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

int main() {
    // extract_images();
    return 0;
}
