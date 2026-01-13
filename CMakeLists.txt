cmake_minimum_required(VERSION 3.10)
project(LoopClosingProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# wildcard
file(GLOB SOURCES "src/*.cpp")

add_executable(LoopClosing ${SOURCES})

target_link_libraries(LoopClosing ${OpenCV_LIBS})