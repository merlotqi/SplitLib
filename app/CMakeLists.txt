add_executable(SplatTransform main.cpp process.h process.cpp gpudevice.h)

target_link_libraries(SplatTransform PRIVATE absl::flags_parse absl::flags splat)
