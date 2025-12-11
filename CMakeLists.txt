cmake_minimum_required(VERSION 3.10)
project(splat-transform LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_ARCHITECTURES "75;80;89" CACHE STRING "CUDA architectures")

find_package(CUDA REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(WebP REQUIRED)
find_package(nlohmann_json REQUIRED)

include(GenerateExportHeader)

add_library(splat SHARED)

add_subdirectory(include)
add_subdirectory(src)

target_link_libraries(splat Eigen3::Eigen ${WebP_LIBRARIES} nlohmann_json::nlohmann_json cuda_kernels)
target_include_directories(splat PUBLIC ${WebP_INCLUDE_DIRS})
if(WIN32)
    target_compile_definitions(splat PUBLIC _USE_MATH_DEFINES)
endif()

add_subdirectory(app)
