#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for STL container support
#include "src/matrix.hpp"

namespace py = pybind11;

template<typename T>
void bind_matrix(py::module &m, const std::string &type_name) {
    py::class_<Matrix<T>>(m, type_name.c_str())
        .def(py::init<uint32_t, uint32_t>(),
            "Initialize matrix with given rows and columns",
            py::arg("rows"), py::arg("cols"))
        .def(py::init<const std::string&>(),
            "Initialize matrix from string representation",
            py::arg("string_repr"))
        .def("export_string", &Matrix<T>::export_string, "Export matrix to string")
        .def("zeros", &Matrix<T>::zeros, "Set all elements to zero")
        .def("identity", &Matrix<T>::identity, "Set matrix to identity matrix")
        .def("fill", &Matrix<T>::fill, "Fill matrix with a specific value")
        .def("randomize", static_cast<Matrix<T>&(Matrix<T>::*)()>(&Matrix<T>::randomize), "Randomize matrix values")
        .def("randomize", static_cast<Matrix<T>&(Matrix<T>::*)(T, T)>(&Matrix<T>::randomize), "Randomize matrix values")
        .def("transpose_inplace", &Matrix<T>::transpose_inplace, "Transpose the matrix")
        .def("as_transposed", &Matrix<T>::as_transposed, "Return a transposed matrix")
        .def("xavier_init", &Matrix<T>::xavier_init, "Initialize matrix with Xavier initialization")
        .def("__add__", static_cast<Matrix<T>(Matrix<T>::*)(T) const>(&Matrix<T>::operator+), "Add two matrices")
        .def("__add__", static_cast<Matrix<T>(Matrix<T>::*)(const Matrix<T>&) const>(&Matrix<T>::operator+), "Add two matrices")
        .def("__sub__", static_cast<Matrix<T>(Matrix<T>::*)(T) const>(&Matrix<T>::operator-), "Sub two matrices")
        .def("__sub__", static_cast<Matrix<T>(Matrix<T>::*)(const Matrix<T>&) const>(&Matrix<T>::operator-), "Add two matrices")
        .def("__mul__", static_cast<Matrix<T>(Matrix<T>::*)(T) const>(&Matrix<T>::operator*), "Multiply two matrices")
        .def("__mul__", static_cast<Matrix<T>(Matrix<T>::*)(const Matrix<T>&) const>(&Matrix<T>::operator*), "Multiply two matrices")
        .def("__div__", &Matrix<T>::operator/, "Divide by scalar")
        .def("__iadd__", static_cast<Matrix<T>&(Matrix<T>::*)(T)>(&Matrix<T>::operator+=), "Add two matrices")
        .def("__iadd__", static_cast<Matrix<T>&(Matrix<T>::*)(const Matrix<T>&)>(&Matrix<T>::operator+=), "Add a scalar")
        .def("__isub__", static_cast<Matrix<T>&(Matrix<T>::*)(T)>(&Matrix<T>::operator-=), "Sub two matrices")
        .def("__isub__", static_cast<Matrix<T>&(Matrix<T>::*)(const Matrix<T>&)>(&Matrix<T>::operator-=), "Sub a scalar")
        .def("__imul__", static_cast<Matrix<T>&(Matrix<T>::*)(T)>(&Matrix<T>::operator*=), "Multiply and assign")
        .def("__imul__", static_cast<Matrix<T>&(Matrix<T>::*)(const Matrix<T>&)>(&Matrix<T>::operator*=), "Multiply and assign")
        .def("__idiv__", &Matrix<T>::operator/=, "Divide and assign")
        .def("__assign__", &Matrix<T>::operator=, "Assign one matrix to another")
        .def("__getitem__", static_cast<T&(Matrix<T>::*)(uint32_t)>(&Matrix<T>::operator[]), "Access matrix element by index")
        .def("__getitem__", static_cast<T(Matrix<T>::*)(uint32_t)const>(&Matrix<T>::operator[]), "Access matrix element by index")
        .def("__len__", &Matrix<T>::size, "Get the total number of elements")
        .def("set_at", &Matrix<T>::set_at, "Set matrix element at given row and column")
        .def("set_at_flat", &Matrix<T>::set_at_flat, "Set matrix element at given flat index")
        .def("get_at", &Matrix<T>::get_at, "Get matrix element at given row and column")
        .def("get_at_flat", &Matrix<T>::get_at_flat, "Get matrix element at given flat index")
        .def("data", &Matrix<T>::data, "Access raw matrix data")
        .def("rows", &Matrix<T>::rows, "Get the number of rows")
        .def("cols", &Matrix<T>::cols, "Get the number of columns")
        .def("size", &Matrix<T>::size, "Get the total number of elements");
}
