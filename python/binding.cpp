#include <pybind11/pybind11.h>

int add(int i, int j) {
  return i + j;
}

std::string hello() {
  return "Hello from pybind11!";
}

PYBIND11_MODULE(splat_transform_cpp, m) {
  m.doc() = "python11 example plugin";

  m.def("add", &add, "a function that add two numbers");
  m.def("hello", &hello, "A function that returns a greeting");
}
