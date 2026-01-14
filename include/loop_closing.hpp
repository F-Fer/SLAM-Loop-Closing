#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <map>

namespace loop_closing {

// Structure to hold frame data
struct Frame {
    int id;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    cv::Mat pose;  // 4x4 transformation matrix
    std::vector<cv::Point3f> points3D;  // 3D points from triangulation
};

// Structure to hold match information
struct LoopCandidate {
    int current_frame_id;
    int matched_frame_id;
    int num_matches;
    double similarity_score;
};

class LoopClosingSystem {
public:
    LoopClosingSystem(double loop_threshold = 0.7, int min_loop_gap = 30);
    
    // Process a single frame
    void processFrame(const cv::Mat& image, int frame_id);
    
    // Detect features in an image
    void detectFeatures(Frame& frame);
    
    // Match features between two frames
    std::vector<cv::DMatch> matchFeatures(const Frame& frame1, const Frame& frame2);
    
    // Estimate relative pose between two frames
    bool estimatePose(const Frame& frame1, const Frame& frame2, 
                      const std::vector<cv::DMatch>& matches,
                      cv::Mat& R, cv::Mat& t);
    
    // Check for loop closure
    std::vector<LoopCandidate> detectLoops(int current_frame_id);
    
    // Triangulate points from two views
    void triangulatePoints(const Frame& frame1, const Frame& frame2,
                          const std::vector<cv::DMatch>& matches,
                          std::vector<cv::Point3f>& points3D);
    
    // Visualize matches between two frames
    cv::Mat visualizeMatches(const Frame& frame1, const Frame& frame2,
                            const std::vector<cv::DMatch>& matches);
    
    // Get all processed frames
    const std::vector<Frame>& getFrames() const { return frames_; }
    
    // Get detected loop closures
    const std::vector<LoopCandidate>& getLoopClosures() const { return loop_closures_; }
    
    // Save results to file
    void saveResults(const std::string& output_dir);

private:
    std::vector<Frame> frames_;
    std::vector<LoopCandidate> loop_closures_;
    
    cv::Ptr<cv::ORB> feature_detector_;
    cv::Ptr<cv::BFMatcher> matcher_;
    
    double loop_threshold_;    // Similarity threshold for loop detection
    int min_loop_gap_;         // Minimum frame gap for loop detection
    
    // Camera intrinsics (assuming a typical camera)
    cv::Mat K_;  // Camera matrix
};

} // namespace loop_closing
