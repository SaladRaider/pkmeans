cmake_minimum_required(VERSION 2.8)
find_package (Threads)

# build the library sources from all sources
file(GLOB lib_sources *.cpp)

# pkmeans library
add_library(pkmeans_lib ${lib_sources})
