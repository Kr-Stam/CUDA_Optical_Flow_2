cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

if (MSVC)
  set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0/bin/nvcc.exe")
  set(CMAKE_CXX_COMPILER "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64/cl.exe")
endif()

if (UNIX)
		message(STATUS "Project is building for Linux")
    set(CMAKE_CUDA_COMPILER /opt/cuda/bin/nvcc)
		set(CMAKE_CXX_STANDARD 17)
		set(CMAKE_CUDA_STANDARD 17)

		set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

		message(STATUS "Importing opencv")
		find_package( OpenCV REQUIRED )
		include_directories(
			"${CMAKE_SOURCE_DIR}"
			"${OpenCV_INCLUDE_DIRS}"
		)
endif()

if(WIN32)
    message(STATUS "Project is building for Windows")
    set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3")
    set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3/bin/nvcc.exe")
		set(CMAKE_CXX_STANDARD 17)
		set(CMAKE_CUDA_STANDARD 17)
		set(DCMAKE_CUDA_ARCHITECTURES native)

		message(STATUS "Importing opencv")
		set("OpenCV_DIR" "C:/opencv/build")
		find_package( OpenCV REQUIRED )
		include_directories(
			"${CMAKE_SOURCE_DIR}"
			"${CMAKE_CURRENT_SOURCE_DIR}"
			"${OpenCV_INCLUDE_DIRS}"
		)
endif()


if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	message(STATUS "Building in Debug mode")
endif()

if(MNIST)
	project(MNIST LANGUAGES CXX CUDA)
elseif(TEST_ALL)
	project(Test_All LANGUAGES CXX CUDA)
else()
	project(CUDA_Optical_Flow LANGUAGES CXX CUDA)
endif()

message("Project name: " ${CMAKE_PROJECT_NAME})


if (MSVC)
	message(STATUS "Building for Windows with MSVC")
	set(_CRT_SECURE_NO_WARNINGS 1)
	set(CURL_STATICLIB 1)
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

file(GLOB SOURCE_FILES_CPP "*.cpp")
file(GLOB SOURCE_FILES_CU "*.cu")
add_executable(${CMAKE_PROJECT_NAME} ${SOURCE_FILES_CPP} ${SOURCE_FILES_CU} main.cu)
target_link_libraries( 
	"${CMAKE_PROJECT_NAME}"
	"${OpenCV_LIBS}"
)

target_compile_options(${CMAKE_PROJECT_NAME}
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>: >
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored> # -g -G --ptxas-options=–allow-expensive-optimizations=true>
)

set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET ${CMAKE_PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 61 75 87 80) # Running on GTX 1080 Ti, GTX 1660 Ti, and Jetson AGX Orin

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE _DEBUG)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()
