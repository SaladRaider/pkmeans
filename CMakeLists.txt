cmake_minimum_required(VERSION 3.5)

project(pkmeans)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1z")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")

# Set up gtest. This must be set up before any subdirectories are
# added which will use gtest.
add_subdirectory(extern/googletest)
find_library(gtest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})
link_directories(${GTEST_LIBS_DIR})
add_definitions("-fPIC")
enable_testing()

add_compile_options(-D_FORTIFY_SOURCE=2 -D_GLBICXX_ASSERTIONS
    -fasynchronous-unwind-tables -fexceptions -fpie -fpic -fstack-protector -grecord-gcc-switches -O3 -pipe -std=c++17)

#
# set up boost
#
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
find_package(Boost COMPONENTS program_options REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIRS})
message(STATUS "Boost Include: ${Boost_INCLUDE_DIR}")
message(STATUS "Boost Libraries: ${Boost_LIBRARY_DIRS}")
message(STATUS "Boost Libraries: ${Boost_LIBRARIES}")

# Also search for includes in PROJECT_BINARY_DIR to find config.h.
include_directories("${PROJECT_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_subdirectory(src)
