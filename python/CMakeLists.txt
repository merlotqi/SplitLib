pybind11_add_module(splat_transform_cpp binding.cpp)

target_link_libraries(splat_transform_cpp PRIVATE SPLAT::splat)

set_target_properties(splat_transform_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python_lib"
)
