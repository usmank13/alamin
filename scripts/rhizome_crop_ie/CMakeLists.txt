cmake_minimum_required(VERSION 3.20)
project(alamin_rhizome_crop_ie LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(RHIZOME_ROOT "" CACHE PATH "Rhizome checkout with src/e2e")
set(RHIZOME_HOST_BUILD "" CACHE PATH "Matching host build containing lib/librhizome.a")

# Build the GPU implementations from the selected Rhizome checkout.
set(E2E_PIPELINE ON CACHE BOOL "" FORCE)
set(E2E_TORCH ON CACHE BOOL "" FORCE)
set(E2E_TORCH_CUDA ON CACHE BOOL "" FORCE)
set(E2E_TRT ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
add_subdirectory("${RHIZOME_ROOT}/src/e2e" runtime)
find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)
find_package(ZLIB REQUIRED)
find_library(RHIZOME_ARCHIVE rhizome PATHS "${RHIZOME_HOST_BUILD}/lib" REQUIRED NO_DEFAULT_PATH)
find_library(TORCH_CUDA_LIB torch_cuda PATHS "${E2E_TORCH_ROOT}/lib" REQUIRED NO_DEFAULT_PATH)
find_path(GST_INCLUDE_DIR gst/gst.h PATH_SUFFIXES gstreamer-1.0 REQUIRED)
find_path(GLIB_INCLUDE_DIR glib.h PATH_SUFFIXES glib-2.0 REQUIRED)
find_path(GLIB_CONFIG_DIR glibconfig.h PATHS "/usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}" PATH_SUFFIXES glib-2.0/include REQUIRED)
find_path(OPENCV_INCLUDE_DIR opencv2/core.hpp PATH_SUFFIXES opencv4 REQUIRED)
find_library(GSTREAMER_LIB NAMES gstreamer-1.0 libgstreamer-1.0.so.0 REQUIRED)
find_library(GSTAPP_LIB NAMES gstapp-1.0 libgstapp-1.0.so.0 REQUIRED)
add_executable(crop_ie "${RHIZOME_ROOT}/src/e2e/crop_ie.main.cpp" "${RHIZOME_ROOT}/src/e2e/debug_output.cpp")
target_compile_definitions(crop_ie PRIVATE HOST_BUILD ALLOW_SIMULATION)
target_include_directories(crop_ie PRIVATE "${RHIZOME_ROOT}/src" "${RHIZOME_ROOT}/include"
    "${RHIZOME_ROOT}/external/include" "${RHIZOME_ROOT}/external/box2d/include"
    "${GST_INCLUDE_DIR}" "${GLIB_INCLUDE_DIR}" "${GLIB_CONFIG_DIR}" "${OPENCV_INCLUDE_DIR}")
# Keep CUDA's ATen kernel registrations even when the executable only calls the dispatcher.
target_link_libraries(crop_ie PRIVATE "${RHIZOME_ARCHIVE}" e2e_pipeline e2e_torch_options
    "-Wl,--no-as-needed" "${TORCH_CUDA_LIB}" CUDA::cudart "-Wl,--as-needed"
    "${GSTAPP_LIB}" "${GSTREAMER_LIB}" gobject-2.0 glib-2.0 opencv_core opencv_imgproc
    opencv_imgcodecs zmq systemd ZLIB::ZLIB Threads::Threads ${CMAKE_DL_LIBS})
add_dependencies(crop_ie e2e_trt e2e_torchscript)
