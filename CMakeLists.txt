# CMake 3.25.2 is required for NVCC 12 + CUDA20 support
cmake_minimum_required(VERSION 3.25.2)

project(diffmoments LANGUAGES CXX)

# Nanobind setup from https://github.com/wjakob/nanobind_example/blob/master/CMakeLists.txt
if (NOT SKBUILD)
    message(WARNING "\
    If you are a user trying to install this package, please use the 
    command below, which will install all necessary build dependencies, 
    compile the package in an isolated environment, and then install it.
    =====================================================================
    $ pip install .
    =====================================================================
    If you are a software developer, and this is your own package, then
    it is usually much more efficient to install the build dependencies
    in your environment once and use the following command that avoids
    a costly creation of a new virtual environment at every compilation:
    =====================================================================
    $ pip install scikit-build-core[pyproject]
    $ pip install --no-build-isolation -ve .
    =====================================================================
    You may optionally add -Ceditable.rebuild=true to auto-rebuild when
    the package is imported. Otherwise, you need to re-run the above
    after editing C++ files. If you want to modify the CUDA kernels,
    add -Ccmake.define.DM_BUILD_KERNELS=ON to the command.")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS FALSE)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(DM_BUILD_KERNELS "Build the CUDA kernels" OFF)
option(DM_BUILD_TESTS "Build tests" OFF)

# Try to import all Python components potentially needed by nanobind
find_package(Python 3.8
    REQUIRED COMPONENTS Interpreter Development.Module
    OPTIONAL_COMPONENTS Development.SABIModule
)

add_subdirectory(ext/nanobind)

add_subdirectory(src)