// pywrap.cpp
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>

#include "voromesh.h"
#include "self_intersect.h"

namespace py = pybind11;


PYBIND11_MODULE(VoroMeshUtils, m) {
    m.doc() = "optional module docstring";

    // py::module m("VoroMeshUtils", "pybind11 plugin");
    m.def("compute_voromesh", &compute_voromesh, "compute the voromesh from given positions and values");
    m.def("self_intersect", &self_intersect, "compute the number of self_intersection from a given file");
    // return m.ptr();
}

