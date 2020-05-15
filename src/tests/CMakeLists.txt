include_directories(tests/)

# build the test
file(GLOB test_sources *.test.cpp)
add_executable(pkmeans_tests ${test_sources})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pkmeans_tests PUBLIC OpenMP::OpenMP_CXX PRIVATE pkmeans_lib gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
else()
    target_link_libraries(pkmeans_tests pkmeans_lib gtest gtest_main ${CMAKE_THREAD_LIBS_INIT})
endif()

add_test(TestPKMeans ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/pkmeans_tests)
