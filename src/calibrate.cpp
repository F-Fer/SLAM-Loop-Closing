#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main() {
    // Chessboard dimensions (number of internal corners)
    // For a board with 10x7 squares, there are 9x6 internal corners
    const int chessboardRows = 6;
    const int chessboardCols = 9;
    const float squareSize = 3.0; // Size of a chessboard square in cm (e.g., 3.0 for 3cm)

    // Prepare object points for the chessboard
    std::vector<cv::Point3f> objPts;
    for (int i = 0; i < chessboardRows; ++i) {
        for (int j = 0; j < chessboardCols; ++j) {
            objPts.emplace_back(j * squareSize, i * squareSize, 0);
        }
    }

    // Vectors to store 2D image points and 3D object points for each image
    std::vector<std::vector<cv::Point3f>> objectPoints;
    std::vector<std::vector<cv::Point2f>> imagePoints;

    // Load calibration images from the directory
    std::string path = "../data/calibration/*.jpeg";
    std::vector<cv::String> imageFiles;
    cv::glob(path, imageFiles);

    if (imageFiles.empty()) {
        std::cerr << "No images found in " << path << std::endl;
        return -1;
    }

    std::cout << "Found " << imageFiles.size() << " images." << std::endl;

    int rows = 0, cols = 0;

    for (const auto& file : imageFiles) {
        cv::Mat img = cv::imread(file);
        if (img.empty()) {
            std::cerr << "Error loading image: " << file << ". Check if file is a valid image." << std::endl;
            continue;
        }

        std::cout << "Processing: " << file << " (" << img.cols << "x" << img.rows << ")" << std::endl;
        
        // Save image size for calibration
        cols = img.cols;
        rows = img.rows;

        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

        std::vector<cv::Point2f> corners;
        
        // Try multiple sizes and resolutions for robustness
        std::vector<cv::Size> boardSizes = {
            cv::Size(chessboardCols, chessboardRows),
            cv::Size(chessboardRows, chessboardCols)
        };

        bool found = false;
        
        // Try at original resolution first, then at half resolution
        for (int scale = 1; scale <= 2; ++scale) {
            cv::Mat detectionImg;
            if (scale == 1) {
                detectionImg = gray;
            } else {
                cv::resize(gray, detectionImg, cv::Size(), 1.0/scale, 1.0/scale);
            }

            for (const auto& boardSize : boardSizes) {
                // Using findChessboardCornersSB which is more robust
                found = cv::findChessboardCornersSB(detectionImg, boardSize, corners,
                    cv::CALIB_CB_ACCURACY | cv::CALIB_CB_EXHAUSTIVE);
                
                if (found) {
                    if (scale > 1) {
                        // Scale corners back to original resolution
                        for (auto& corner : corners) {
                            corner.x *= scale;
                            corner.y *= scale;
                        }
                    }
                    
                    // Refine on original gray image
                    cv::cornerSubPix(gray, corners, cv::Size(11, 11), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
                    
                    // Add points based on found orientation
                    imagePoints.push_back(corners);
                    if (boardSize == cv::Size(chessboardCols, chessboardRows)) {
                        objectPoints.push_back(objPts);
                    } else {
                        std::vector<cv::Point3f> objPtsRotated;
                        for (int i = 0; i < chessboardCols; ++i) {
                            for (int j = 0; j < chessboardRows; ++j) {
                                objPtsRotated.emplace_back(i * squareSize, j * squareSize, 0);
                            }
                        }
                        objectPoints.push_back(objPtsRotated);
                    }
                    break;
                }
            }
            if (found) break;
        }

        if (found) {
            std::cout << "Successfully found corners in " << file << std::endl;

            // Draw and display the corners
            cv::Mat drawImg = img.clone();
            cv::drawChessboardCorners(drawImg, 
                corners.size() == (size_t)(chessboardCols * chessboardRows) ? cv::Size(chessboardCols, chessboardRows) : cv::Size(chessboardRows, chessboardCols), 
                corners, found);

            // downsize image for display
            cv::Size displaySize(cols / 4, rows / 4);
            cv::Mat resizedImage;
            cv::resize(drawImg, resizedImage, displaySize, 0, 0, cv::INTER_LINEAR);

            cv::imshow("Corners", resizedImage);
            cv::waitKey(500); 
        }
        else {
            std::cerr << "Chessboard corners not found in image: " << file << std::endl;
        }
    }
    cv::destroyAllWindows();

    // Camera calibration
    if (imagePoints.empty()) {
        std::cerr << "Error: No corners found in any image. Calibration failed." << std::endl;
        return -1;
    }

    cv::Mat cameraMatrix, distCoeffs, R, T;
    std::vector<cv::Mat> rvecs, tvecs;
    double rms = cv::calibrateCamera(objectPoints, imagePoints, cv::Size(cols, rows),
        cameraMatrix, distCoeffs, rvecs, tvecs);

    // Output the calibration results
    std::cout << "Reprojection error: " << rms << std::endl;
    std::cout << "Camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients:\n" << distCoeffs << std::endl;

    return 0;
}
