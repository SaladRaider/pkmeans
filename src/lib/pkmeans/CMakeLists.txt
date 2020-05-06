cmake_minimum_required(VERSION 2.8)
find_package (Threads)

# build the library sources from all sources
file(GLOB lib_sources *.cpp *.hpp)

# pkmeans library
add_library(pkmeans_lib ${lib_sources})
target_link_libraries(pkmeans_lib gtest gtest_main)
