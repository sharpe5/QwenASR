cmake_minimum_required(VERSION 3.10)
set(PROJECT_NAME "qwen_asr")
project(${PROJECT_NAME} LANGUAGES CXX)

set(PLUGIN_NAME "qwen_asr_plugin")

add_library(${PLUGIN_NAME} SHARED
  "qwen_asr_plugin.cc"
  "qwen_asr_plugin.h"
)

apply_standard_settings(${PLUGIN_NAME})
set_target_properties(${PLUGIN_NAME} PROPERTIES
  CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${PLUGIN_NAME} PRIVATE FLUTTER_PLUGIN_IMPL)

target_include_directories(${PLUGIN_NAME} INTERFACE
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

target_link_libraries(${PLUGIN_NAME} PRIVATE flutter)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GTK REQUIRED IMPORTED_TARGET gtk+-3.0)
target_link_libraries(${PLUGIN_NAME} PRIVATE PkgConfig::GTK)

# Include the Rust library via cargokit
include("${CMAKE_CURRENT_SOURCE_DIR}/../cargokit/cmake/cargokit.cmake")
apply_cargokit(${PLUGIN_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/../rust rust_lib_qwen_asr "")
