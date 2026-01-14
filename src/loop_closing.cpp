#include "loop_closing.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

namespace loop_closing {

LoopClosingSystem::LoopClosingSystem(double loop_threshold, int min_loop_gap)
    : loop_threshold_(loop_threshold), min_loop_gap_(min_loop_gap) {
    
    // Initialize ORB feature detector
    feature_detector_ = cv::ORB::create(2000);  // Detect up to 2000 features
    
    // Initialize BFMatcher with Hamming distance for ORB
    matcher_ = cv::BFMatcher::create(cv::NORM_HAMMING, true);
    
    // Initialize camera matrix with typical values (adjust based on your camera)
    K_ = (cv::Mat_<double>(3, 3) << 
        800, 0, 640,    // fx, 0, cx
        0, 800, 360,    // 0, fy, cy
        0, 0, 1);       // 0, 0, 1
}

void LoopClosingSystem::detectFeatures(Frame& frame) {
    // Convert to grayscale if needed
    cv::Mat gray;
    if (frame.image.channels() == 3) {
        cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = frame.image;
    }
    
    // Detect keypoints and compute descriptors
    feature_detector_->detectAndCompute(gray, cv::noArray(), 
                                        frame.keypoints, frame.descriptors);
    
    std::cout << "Frame " << frame.id << ": Detected " << frame.keypoints.size() 
              << " features" << std::endl;
}

std::vector<cv::DMatch> LoopClosingSystem::matchFeatures(const Frame& frame1, 
                                                         const Frame& frame2) {
    if (frame1.descriptors.empty() || frame2.descriptors.empty()) {
        return std::vector<cv::DMatch>();
    }
    
    std::vector<cv::DMatch> matches;
    matcher_->match(frame1.descriptors, frame2.descriptors, matches);
    
    // Filter matches using distance threshold
    if (matches.empty()) {
        return matches;
    }
    
    double max_dist = 0;
    double min_dist = 100;
    
    for (const auto& match : matches) {
        double dist = match.distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }
    
    // Keep only good matches (distance < 3 * min_dist)
    std::vector<cv::DMatch> good_matches;
    double threshold = std::max(2.0 * min_dist, 30.0);
    
    for (const auto& match : matches) {
        if (match.distance <= threshold) {
            good_matches.push_back(match);
        }
    }
    
    return good_matches;
}

bool LoopClosingSystem::estimatePose(const Frame& frame1, const Frame& frame2,
                                     const std::vector<cv::DMatch>& matches,
                                     cv::Mat& R, cv::Mat& t) {
    if (matches.size() < 8) {
        return false;  // Need at least 8 points for essential matrix
    }
    
    // Extract matched points
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(frame1.keypoints[match.queryIdx].pt);
        points2.push_back(frame2.keypoints[match.trainIdx].pt);
    }
    
    // Find essential matrix
    cv::Mat E, mask;
    E = cv::findEssentialMat(points1, points2, K_, cv::RANSAC, 0.999, 1.0, mask);
    
    if (E.empty()) {
        return false;
    }
    
    // Recover pose from essential matrix
    int inliers = cv::recoverPose(E, points1, points2, K_, R, t, mask);
    
    // Check if we have enough inliers
    return inliers >= 8;
}

void LoopClosingSystem::triangulatePoints(const Frame& frame1, const Frame& frame2,
                                          const std::vector<cv::DMatch>& matches,
                                          std::vector<cv::Point3f>& points3D) {
    if (matches.size() < 8) {
        return;
    }
    
    // Extract matched points
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(frame1.keypoints[match.queryIdx].pt);
        points2.push_back(frame2.keypoints[match.trainIdx].pt);
    }
    
    // Create projection matrices
    cv::Mat P1 = K_ * cv::Mat::eye(3, 4, CV_64F);  // [K | 0]
    
    // For frame2, we need R and t
    cv::Mat R, t;
    if (!estimatePose(frame1, frame2, matches, R, t)) {
        return;
    }
    
    cv::Mat Rt;
    cv::hconcat(R, t, Rt);
    cv::Mat P2 = K_ * Rt;
    
    // Convert to double precision
    P1.convertTo(P1, CV_64F);
    P2.convertTo(P2, CV_64F);
    
    // Triangulate points
    cv::Mat points4D;
    try {
        cv::triangulatePoints(P1, P2, points1, points2, points4D);
        
        // Convert from homogeneous to 3D coordinates
        points3D.clear();
        for (int i = 0; i < points4D.cols; i++) {
            float w = points4D.at<float>(3, i);
            if (std::abs(w) > 1e-6f) {  // Avoid division by zero
                float x = points4D.at<float>(0, i) / w;
                float y = points4D.at<float>(1, i) / w;
                float z = points4D.at<float>(2, i) / w;
                
                // Filter out points that are too far or behind the camera
                if (z > 0 && z < 100) {
                    points3D.push_back(cv::Point3f(x, y, z));
                }
            }
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Triangulation error: " << e.what() << std::endl;
    }
}

void LoopClosingSystem::processFrame(const cv::Mat& image, int frame_id) {
    Frame frame;
    frame.id = frame_id;
    frame.image = image.clone();
    
    // Detect features
    detectFeatures(frame);
    
    // If this is not the first frame, match with previous frame
    if (!frames_.empty()) {
        const Frame& prev_frame = frames_.back();
        auto matches = matchFeatures(prev_frame, frame);
        
        std::cout << "Matches between frame " << prev_frame.id << " and " 
                  << frame.id << ": " << matches.size() << std::endl;
        
        // Estimate pose
        cv::Mat R, t;
        if (estimatePose(prev_frame, frame, matches, R, t)) {
            std::cout << "Pose estimated successfully" << std::endl;
            
            // Triangulate points
            std::vector<cv::Point3f> points3D;
            triangulatePoints(prev_frame, frame, matches, points3D);
            frame.points3D = points3D;
            
            std::cout << "Triangulated " << points3D.size() << " 3D points" << std::endl;
        }
    }
    
    // Check for loop closure
    auto loop_candidates = detectLoops(frame_id);
    if (!loop_candidates.empty()) {
        std::cout << "\n*** Loop closure detected! ***" << std::endl;
        for (const auto& candidate : loop_candidates) {
            std::cout << "  Frame " << candidate.current_frame_id 
                      << " matches with frame " << candidate.matched_frame_id
                      << " (score: " << candidate.similarity_score << ")" << std::endl;
            
            // Perform additional matching with the loop closure frame
            const Frame& loop_frame = frames_[candidate.matched_frame_id];
            auto loop_matches = matchFeatures(loop_frame, frame);
            
            std::cout << "  Additional loop matches: " << loop_matches.size() << std::endl;
            
            // Re-triangulate with loop closure
            std::vector<cv::Point3f> loop_points3D;
            triangulatePoints(loop_frame, frame, loop_matches, loop_points3D);
            std::cout << "  Loop triangulated points: " << loop_points3D.size() << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Add frame to collection
    frames_.push_back(frame);
}

std::vector<LoopCandidate> LoopClosingSystem::detectLoops(int current_frame_id) {
    std::vector<LoopCandidate> candidates;
    
    if (frames_.empty() || current_frame_id < min_loop_gap_) {
        return candidates;
    }
    
    const Frame& current_frame = frames_.back();
    
    // Compare with previous frames (excluding recent ones)
    for (int i = 0; i < static_cast<int>(frames_.size()) - min_loop_gap_; i++) {
        const Frame& candidate_frame = frames_[i];
        
        // Match features
        auto matches = matchFeatures(candidate_frame, current_frame);
        
        if (matches.empty()) {
            continue;
        }
        
        // Calculate similarity score based on number of matches
        double similarity = static_cast<double>(matches.size()) / 
                          std::min(candidate_frame.keypoints.size(), 
                                  current_frame.keypoints.size());
        
        // If similarity exceeds threshold, it's a loop candidate
        if (similarity >= loop_threshold_ && matches.size() >= 50) {
            LoopCandidate loop;
            loop.current_frame_id = current_frame_id;
            loop.matched_frame_id = candidate_frame.id;
            loop.num_matches = matches.size();
            loop.similarity_score = similarity;
            
            candidates.push_back(loop);
            loop_closures_.push_back(loop);
        }
    }
    
    return candidates;
}

cv::Mat LoopClosingSystem::visualizeMatches(const Frame& frame1, const Frame& frame2,
                                            const std::vector<cv::DMatch>& matches) {
    cv::Mat img_matches;
    cv::drawMatches(frame1.image, frame1.keypoints,
                   frame2.image, frame2.keypoints,
                   matches, img_matches);
    return img_matches;
}

void LoopClosingSystem::saveResults(const std::string& output_dir) {
    // Create output directory if it doesn't exist
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
    }
    
    // Save loop closure information
    std::ofstream loop_file(output_dir + "/loop_closures.txt");
    loop_file << "Loop Closures Detected:\n";
    loop_file << "======================\n\n";
    
    for (const auto& loop : loop_closures_) {
        loop_file << "Frame " << loop.current_frame_id 
                  << " <-> Frame " << loop.matched_frame_id << "\n";
        loop_file << "  Matches: " << loop.num_matches << "\n";
        loop_file << "  Similarity: " << loop.similarity_score << "\n\n";
    }
    
    loop_file.close();
    
    // Save visualizations of consecutive frame matches
    std::cout << "Saving match visualizations..." << std::endl;
    for (size_t i = 1; i < frames_.size(); i++) {
        if (i % 10 == 0) {  // Save every 10th frame to avoid too many images
            auto matches = matchFeatures(frames_[i-1], frames_[i]);
            cv::Mat vis = visualizeMatches(frames_[i-1], frames_[i], matches);
            
            std::string filename = output_dir + "/matches_" + 
                                 std::to_string(i-1) + "_" + 
                                 std::to_string(i) + ".png";
            cv::imwrite(filename, vis);
        }
    }
    
    // Save visualizations of loop closures
    std::cout << "Saving loop closure visualizations..." << std::endl;
    for (const auto& loop : loop_closures_) {
        const Frame& frame1 = frames_[loop.matched_frame_id];
        const Frame& frame2 = frames_[loop.current_frame_id];
        
        auto matches = matchFeatures(frame1, frame2);
        cv::Mat vis = visualizeMatches(frame1, frame2, matches);
        
        std::string filename = output_dir + "/loop_" + 
                             std::to_string(loop.matched_frame_id) + "_" + 
                             std::to_string(loop.current_frame_id) + ".png";
        cv::imwrite(filename, vis);
    }
    
    std::cout << "Results saved to: " << output_dir << std::endl;
}

} // namespace loop_closing
