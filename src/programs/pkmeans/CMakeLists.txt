project(pkmeans_executable)

add_executable(pkmeans main.cu)

target_compile_features(pkmeans PUBLIC cxx_std_14)

target_link_libraries(pkmeans
    pkmeans_lib
    ${Boost_LIBRARIES}
)

set_target_properties(pkmeans
    PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

