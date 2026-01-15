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

# wildcard
file(GLOB SOURCES "src/*.cpp")

add_executable(LoopClosing ${SOURCES})

target_link_libraries(LoopClosing ${OpenCV_LIBS})