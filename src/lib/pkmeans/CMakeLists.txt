cmake_minimum_required(VERSION 2.8)
find_package (Threads)

# build the library sources from all sources
file(GLOB lib_sources *.cpp *.hpp *.cu)

# pkmeans library
add_library(pkmeans_lib ${lib_sources})
target_compile_features(pkmeans_lib PUBLIC cxx_std_14)
target_link_libraries(pkmeans_lib gtest gtest_main)
set_target_properties(pkmeans_lib
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
