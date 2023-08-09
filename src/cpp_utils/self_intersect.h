#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <fstream>
#include <Eigen/Dense>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;
// COMPUTES THE NUMBER OF SELF INTERSECTIONS
int self_intersect(char *filename)
{
  Mesh mesh;
  PMP::IO::read_polygon_mesh(filename, mesh);
  if (!CGAL::is_triangle_mesh(mesh))
  {
    std::cerr << "Not a valid input file." << std::endl;
    return 1;
  }

  std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
  PMP::self_intersections(mesh, std::back_inserter(intersected_tris));

  return intersected_tris.size();
}
