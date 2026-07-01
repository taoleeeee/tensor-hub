cmake_minimum_required(VERSION 3.22.1)
project("whisper_native")

# Create the shared library target FIRST
add_library(whisper_native SHARED
    whisper_jni.cpp
    whisper_decoder.cpp
    whisper_mel.cpp
)

target_include_directories(whisper_native PRIVATE .)

# Bypass Prefab find_package - use TFLite headers and libs directly from AAR
# AGP's prefab extraction puts headers and libs in CMAKE_FIND_ROOT_PATH
set(TFLITE_PREFAB_DIR "${CMAKE_FIND_ROOT_PATH}")

target_include_directories(whisper_native PRIVATE
    ${TFLITE_PREFAB_DIR}/include
)

# Find the TFLite shared library
find_library(TFLITE_LIB 
    NAMES tensorflow-lite tensorflowlite
    PATHS ${TFLITE_PREFAB_DIR}/lib/${ANDROID_ABI}
    NO_DEFAULT_PATH
)

if(NOT TFLITE_LIB)
    message(STATUS "TFLite not in Prefab dir, trying Prefab find_package fallback")
    find_package(tensorflowlite QUIET CONFIG)
    if(tensorflowlite_FOUND)
        target_link_libraries(whisper_native tensorflowlite::tensorflowlite log android)
        return()
    endif()
    message(FATAL_ERROR "Could not find TFLite library. Add a TFLite AAR with native libs to your dependencies.")
endif()

message(STATUS "Found TFLite library: ${TFLITE_LIB}")
target_link_libraries(whisper_native ${TFLITE_LIB} log android)
