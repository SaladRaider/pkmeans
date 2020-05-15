project(pkmeans_executable)

add_executable(pkmeans main.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pkmeans PUBLIC OpenMP::OpenMP_CXX PRIVATE pkmeans_lib ${Boost_LIBRARIES})
else()
    target_link_libraries(pkmeans
        pkmeans_lib
        ${Boost_LIBRARIES}
    )
endif()
