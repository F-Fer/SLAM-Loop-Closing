cmake_minimum_required(VERSION 3.10)

# Force system compiler to avoid errors with missing Homebrew GCC
if(APPLE)
    set(CMAKE_C_COMPILER "/usr/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/bin/clang++")
endif()

project(LoopClosingProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Ensure the compiler can find standard headers on macOS
if(APPLE)
    execute_process(COMMAND xcrun --show-sdk-path OUTPUT_VARIABLE SDK_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_OSX_SYSROOT ${SDK_PATH})
    include_directories(${SDK_PATH}/usr/include/c++/v1)
endif()

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

# wildcard
file(GLOB SOURCES "src/*.cpp")

add_executable(LoopClosing ${SOURCES})

target_link_libraries(LoopClosing ${OpenCV_LIBS})