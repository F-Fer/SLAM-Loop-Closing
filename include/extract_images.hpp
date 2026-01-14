#pragma once
#include <string>

/**
 * Extracts frames from a video file and saves them as images.
 * @param video_path Path to the input .MOV file.
 * @param output_dir Directory where the images will be saved.
 * @return true if successful, false otherwise.
 */
bool extract_images_from_mov(const std::string& video_path, const std::string& output_dir);
