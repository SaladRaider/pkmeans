project(pkmeans_executable)

add_executable(pkmeans main.cpp)

target_link_libraries(pkmeans
    pkmeans_lib
    ${Boost_LIBRARIES}
)
