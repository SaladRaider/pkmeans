project(pkmeans)
cmake_minimum_required(VERSION 3.5)

add_compile_options(-D_FORTIFY_SOURCE=2 -D_GLBICXX_ASSERTIONS
    -fasynchronous-unwind-tables -fexceptions -fpie -fpic -fstack-protector -g
    -grecord-gcc-switches -O2 -pipe -std=c++17)

add_definitions("-fPIC")

enable_testing()
add_subdirectory(extern/googletest)

add_library(pkmeans_lib STATIC src/pkmeans.cpp)

add_executable(pkmeans_tests tests/pkmeanstests.cpp)
target_link_libraries(pkmeans_tests PUBLIC
    gtest_main
    pkmeans_lib
)

add_test(
    NAME pkmeans_tests
    COMMAND pkmeans_tests
)

include_directories(src)
file(GLOB SOURCES src/*.h src/*.cpp)
add_executable(pkmeans src/main.cpp src/pkmeans.cpp)
target_compile_features(pkmeans PRIVATE cxx_lambda_init_captures)
