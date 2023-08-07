#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/self_intersections.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Surface_mesh<K::Point_3> Mesh;
typedef boost::graph_traits<Mesh>::face_descriptor face_descriptor;

namespace PMP = CGAL::Polygon_mesh_processing;
// COMPUTES THE NUMBER OF SELF INTERSECTIONS
int main(int argc, char *argv[])
{
  const char *filename = (argc > 1) ? argv[1] : "data/pig.off";

  Mesh mesh;
  PMP::IO::read_polygon_mesh(filename, mesh);
  if (!CGAL::is_triangle_mesh(mesh))
  {
    std::cerr << "Not a valid input file." << std::endl;
    return 1;
  }

  bool intersecting = PMP::does_self_intersect(mesh,
                                               PMP::parameters::vertex_point_map(get(CGAL::vertex_point, mesh)));

  std::vector<std::pair<face_descriptor, face_descriptor>> intersected_tris;
  PMP::self_intersections(mesh, std::back_inserter(intersected_tris));

  std::cout << intersected_tris.size() << std::endl;

  return 0;
}
