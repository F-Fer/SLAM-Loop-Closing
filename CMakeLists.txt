cmake_minimum_required(VERSION 3.10)
project(LoopClosingProject)

# Fix for macOS system headers if they are not found automatically
if(APPLE)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    include_directories(SYSTEM ${SDK_PATH}/usr/include/c++/v1)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# Source files for the main project (excluding calibration)
set(LOOP_CLOSING_SOURCES 
    src/main.cpp 
    src/extract_images_from_mov.cpp
)

add_executable(LoopClosing ${LOOP_CLOSING_SOURCES})

# Separate executable for calibration
add_executable(calibrate src/calibrate.cpp)
target_link_libraries(calibrate ${OpenCV_LIBS})

target_link_libraries(LoopClosing ${OpenCV_LIBS})