include_directories(tests/)

# build the test
file(GLOB test_sources *.test.cpp *.test.cu)
add_executable(pkmeans_tests ${test_sources})
target_compile_features(pkmeans_tests PUBLIC cxx_std_14)
target_link_libraries(pkmeans_tests pkmeans_lib gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
set_target_properties(pkmeans_tests
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_test(TestPKMeans ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pkmeans_tests)
