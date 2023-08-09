cmake_minimum_required(VERSION 3.10)

project(VoroMeshUtils)
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_OSX_ARCHITECTURES "x86_64") # for compilation on mac
set(CMAKE_BUILD_TYPE "Release")
set(PYBIND11_PYTHON_VERSION 3.6)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -fPIC -lmpfr -lgmp")

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(CGAL REQUIRED)

pybind11_add_module(${PROJECT_NAME} pywrap.cpp)

target_compile_definitions(${PROJECT_NAME} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
target_include_directories(${PROJECT_NAME} PRIVATE ${PYBIND11_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen)