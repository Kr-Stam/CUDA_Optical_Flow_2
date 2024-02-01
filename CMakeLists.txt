cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

# if (MSVC)
#   set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
#   set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64/cl.exe")
# endif()

if (UNIX)
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif()

if(WIN32)
    message(STATUS "Project is building for Windows")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")
endif()

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS "Building in Debug mode")
endif()

if(MNIST)
    project(MNIST LANGUAGES CXX CUDA)
elseif(TEST_ALL)
    project(Test_All LANGUAGES CXX CUDA)
else()
    project(CUDA_Machine_Vision LANGUAGES CXX CUDA)
endif()

message("Project name: " ${CMAKE_PROJECT_NAME})

SET("OpenCV_DIR" "C:/opencv/build")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

if (MSVC)
    message(STATUS "Building for Windows with MSVC")
    set(_CRT_SECURE_NO_WARNINGS 1)
    set(CURL_STATICLIB 1)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

add_executable(${CMAKE_PROJECT_NAME} main.cu)

target_link_libraries( ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

if(MNIST)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        MNIST/MNIST_main.cu
        MNIST/MNIST_neural_network.cu
        CUDA_Machine_Vision/Utils/file_writer_neural_net.cu
    )
elseif(TEST_ALL)
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        Tests/reduce.test.cu
    )
else()
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        main.cu
    )
endif()

# target_sources(${CMAKE_PROJECT_NAME} PRIVATE
#     CUDA_Machine_Vision/Utils/error_check_cuda.cu
#     CUDA_Machine_Vision/Utils/gpu_info_print.cu
#     CUDA_Machine_Vision/Utils/cuda_helper.cu
# )

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored --expt-relaxed-constexpr> # -g -G --ptxas-options=–allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 75 87) # Running on GTX 1080 Ti, GTX 1660 Ti, and Jetson AGX Orin

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

if(NN_HALF_FLOAT)
    message(STATUS "Using half precision floating for neural network")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE NN_HALF_FLOAT)
else()
    message(STATUS "Using full precision floating for neural network")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE NN_FULL_FLOAT)
endif()

if(ACTIVATION_NOTHING)
    message(STATUS "Not using an activation function")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ACTIVATION_NOTHING)
elseif(ACTIVATION_MICK_RELU)
    message(STATUS "Using the Mick ReLU activation function")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ACTIVATION_MICK_RELU)
endif()