cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "qwen_asr")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "qwen_asr_plugin")

add_library(${PLUGIN_NAME} SHARED
  "qwen_asr_plugin_c_api.cpp"
  "qwen_asr_plugin_c_api.h"
)

apply_standard_settings(${PLUGIN_NAME})
set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${PLUGIN_NAME} PRIVATE flutter flutter_wrapper_plugin)

# Include the Rust library via cargokit
include("${CMAKE_CURRENT_SOURCE_DIR}/../cargokit/cmake/cargokit.cmake")
apply_cargokit(${PLUGIN_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../rust rust_lib_qwen_asr "")
