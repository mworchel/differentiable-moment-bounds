# Add the core library containing the algorithms
add_library(diffmoments INTERFACE common.hpp linalg.hpp matrix.hpp moment_problem.hpp polynomial.hpp)
target_include_directories(diffmoments INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# Add a library for running the core functions on arrays
add_library(diffmoments_dispatch dispatch.hpp kernels.hpp kernel_flags.hpp cuda_driver.hpp cuda_driver.cpp cuda_module.hpp cuda_module.cpp log.hpp log.cpp)
target_link_libraries(diffmoments_dispatch PRIVATE diffmoments)

set(PTX_FILE_IN_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/generated/kernels.ptx)

# Compile the CUDA kernels to PTX as part of the build process (optional)
if (DM_BUILD_KERNELS)
    include(CheckLanguage)
    check_language(CUDA)

    if (CMAKE_CUDA_COMPILER)
        message(STATUS "Compiling CUDA kernels as part of the build process.")
        
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 20)
        set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

        # Compile kernels to PTX
        add_library(diffmoments_kernels OBJECT kernels.cu kernels.hpp kernel_flags.hpp)
        set_target_properties(diffmoments_kernels PROPERTIES CUDA_ARCHITECTURES "60" CUDA_PTX_COMPILATION ON)
        target_compile_definitions(diffmoments_kernels PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/MP>)
        target_link_libraries(diffmoments_kernels PRIVATE diffmoments)
        #target_compile_options(diffmoments_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v -lineinfo -src-in-ptx>) 
        target_compile_options(diffmoments_kernels PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>) 

        # Overwrite the generated PTX file in the source directory
        add_custom_command(
            OUTPUT ${PTX_FILE_IN_SOURCE}
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:diffmoments_kernels> ${PTX_FILE_IN_SOURCE}
            DEPENDS diffmoments_kernels $<TARGET_OBJECTS:diffmoments_kernels>
        )
    else()
        message(WARNING "Cannot compile kernels because CUDA is unavailable.")
    endif()
else()
    message(STATUS "Using precompiled CUDA kernels")
endif()

# Embed the PTX source in a header file and add a dependency to it using the `ptx2h` dummy target
add_executable(bin2c ../tools/bin2c.cpp)
add_custom_command(
    OUTPUT "${PTX_FILE_IN_SOURCE}.h"
    COMMAND bin2c ${PTX_FILE_IN_SOURCE} "${PTX_FILE_IN_SOURCE}.h" ptx_bytes
    DEPENDS bin2c ${PTX_FILE_IN_SOURCE}
    COMMENT "Converting PTX to header file."
)
add_custom_target(ptx2h DEPENDS "${PTX_FILE_IN_SOURCE}.h")
add_dependencies(diffmoments_dispatch ptx2h)

# Add the Python extension
nanobind_add_module(
  diffmoments_ext

  # Target the stable ABI for Python 3.12+, which reduces
  # the number of binary wheels that must be built. This
  # does nothing on older Python versions
  STABLE_ABI

  # Build libnanobind statically and merge it into the
  # extension (which itself remains a shared library)
  #
  # If your project builds multiple extensions, you can
  # replace this flag by NB_SHARED to conserve space by
  # reusing a shared libnanobind across libraries
  NB_STATIC

  # Source code goes here
  ext.cpp
)

target_link_libraries(diffmoments_ext PRIVATE diffmoments diffmoments_dispatch)

# Install directive for scikit-build-core
install(TARGETS diffmoments_ext LIBRARY DESTINATION .)

# Add tests and playgrounds
if (DM_BUILD_TESTS)
    message(STATUS "Building tests")

    # Very rudimentary "tests" for the numerical procedures
    add_executable(tests test.cpp)
    target_link_libraries(tests PRIVATE diffmoments)
    target_compile_definitions(tests PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/MP>)
    
    if (CMAKE_CUDA_COMPILER)
        add_executable(test_dispatch test_ext_cuda.cpp)
        target_link_libraries(test_dispatch PRIVATE diffmoments diffmoments_dispatch)
    endif()
else()
    message(STATUS "Skipping tests")
endif()