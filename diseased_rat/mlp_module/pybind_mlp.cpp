#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for STL container support
#include "src/mlp.hpp"

namespace py = pybind11;

template<typename T>
void bind_mlp(py::module &m, const std::string &type_name) {
    py::class_<Mlp<T>>(m, type_name.c_str())
        .def(py::init<uint32_t, uint32_t, float, const std::vector<std::pair<uint32_t, uint32_t>>&>(),
            "Initialize Mlp with input size, output size, learning rate and hidden layers",
            py::arg("input_size"), py::arg("output_size"), py::arg("learning_rate"), py::arg("hidden_layers"))
        .def(py::init<const std::string&>(), "Load Mlp from file",
            py::arg("filename"))
        .def("save_model", &Mlp<T>::save_model, "Save the model to a file",
            py::arg("filename"))
        .def("train", &Mlp<T>::train, "Train the Mlp with given inputs and labels",
            py::arg("epochs"), py::arg("data"), py::arg("track_loss") = false)
        .def("train_from_file", &Mlp<T>::train_from_file, "Train the Mlp with inputs and labels from a file",
            py::arg("epochs"), py::arg("filename"))
        .def("predict", &Mlp<T>::predict, "Predict the output for given input")
        .def("forward", &Mlp<T>::forward, "Forward pass for given input")
        .def("backward", &Mlp<T>::backward, "Backward pass for given input and labels")
        .def("test_from_file", &Mlp<T>::test_from_file, "Test the model with inputs and labels from a file",
            py::arg("filename"))
        ;
}

PYBIND11_MODULE(_mlp_module, m) {
    bind_mlp<float>(m, "Mlp");
}
