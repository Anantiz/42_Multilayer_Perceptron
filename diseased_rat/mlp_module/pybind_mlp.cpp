#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for STL container support
#include "src/mlp.hpp"

namespace py = pybind11;

template<typename T>
void bind_mlp(py::module &m, const std::string &type_name) {
    using TrainingReport = typename Mlp<T>::t_training_report;

    py::class_<Mlp<T>>(m, type_name.c_str())
        .def(py::init<uint32_t, uint32_t, float, const std::vector<std::pair<uint32_t, uint32_t>>&>(),
            "Initialize Mlp with input size, output size, learning rate and hidden layers",
            py::arg("input_size"), py::arg("output_size"), py::arg("learning_rate"), py::arg("hidden_layers"))
        .def(py::init<const std::string&>(), "Load Mlp from file",
            py::arg("filename"))
        .def("save_model", &Mlp<T>::save_model, "Save the model to a file",
            py::arg("filename"))
        .def("train", &Mlp<T>::train, "Train the Mlp with given inputs and labels",
            py::arg("epochs"), py::arg("data"))
        .def("train_from_file", &Mlp<T>::train_from_file, "Train the Mlp with inputs and labels from a file",
            py::arg("epochs"), py::arg("filename"))
        .def("predict", &Mlp<T>::predict, "Predict the output for given input")
        .def("forward", &Mlp<T>::forward, "Forward pass for given input")
        .def("backward", &Mlp<T>::backward, "Backward pass for given input and labels")
        .def("test_from_file", &Mlp<T>::test_from_file, "Test the model with inputs and labels from a file",
            py::arg("filename"))
        .def("train_test_earlystop", &Mlp<T>::train_test_earlystop, "Train and test the model with inputs and labels from files",
            py::arg("train_file"), py::arg("test_file"), py::arg("epochs"))
        ;

        py::class_<TrainingReport>(m, "TrainingReport")
        .def_readonly("epochs", &TrainingReport::epochs)
        .def_readonly("test_loss", &TrainingReport::test_loss)
        .def_readonly("test_accuracy", &TrainingReport::test_accuracy)
        .def_readonly("train_loss", &TrainingReport::train_loss)
        .def_readonly("train_accuracy", &TrainingReport::train_accuracy);
        m.attr(type_name.c_str()).attr("TrainingReport") = m.attr("TrainingReport");

    py::enum_<typename Mlp<T>::e_activation_method>(m, "ActivationMethod")
        .value("RELU", Mlp<T>::RELU)
        .value("SIGMOID", Mlp<T>::SIGMOID)
        .value("SOFTMAX", Mlp<T>::SOFTMAX)
        .export_values();
}

PYBIND11_MODULE(_mlp_module, m) {
    bind_mlp<float>(m, "Mlp");
}
