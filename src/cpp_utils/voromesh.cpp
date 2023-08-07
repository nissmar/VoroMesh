#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Exact_predicates_exact_constructions_kernel.h>

#include <CGAL/Delaunay_triangulation_3.h>

#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Regular_triangulation_euclidean_traits_3.h>
#include <CGAL/Cartesian_converter.h>
#include <CGAL/MP_Float.h>
#include <CGAL/circulator.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip> // std::setprecision()

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_3<bool, K> Vb;
typedef CGAL::Delaunay_triangulation_cell_base_3<K> Cb;
typedef CGAL::Triangulation_data_structure_3<Vb, Cb> Tds;
typedef CGAL::Delaunay_triangulation_3<K, Tds> DTriangulation;

typedef DTriangulation::Cell_handle Cell_handle;
typedef DTriangulation::Vertex_handle Vertex_handle;
typedef DTriangulation::Finite_edges_iterator Edge_iterator;
typedef DTriangulation::Point Point;
typedef DTriangulation::Cell_circulator Cell_circulator;

typedef std::vector<Point> PolyVertices;
typedef std::vector<int> PolyFace;

Point exact_dual(const DTriangulation &T, Cell_handle cell)
{
  // return T.dual(cell); //inexact dual
  const Point &p0 = cell->vertex(0)->point();
  const Point &p1 = cell->vertex(1)->point();
  const Point &p2 = cell->vertex(2)->point();
  const Point &p3 = cell->vertex(3)->point();

  // exact computation business
  typedef typename CGAL::Exact_predicates_exact_constructions_kernel EK2;
  typedef typename CGAL::Regular_triangulation_euclidean_traits_3<EK2> EK;
  typedef typename CGAL::Cartesian_converter<K, EK> IK_to_EK;
  typedef typename CGAL::Cartesian_converter<EK, K> EK_to_IK;

  IK_to_EK to_exact;
  EK_to_IK to_inexact;
  // return Triangulationcc;

  EK::Construct_circumcenter_3 exact_circumcenter =
      EK().construct_circumcenter_3_object();
  EK::Point_3 ep0(to_exact(p0));
  EK::Point_3 ep1(to_exact(p1));
  EK::Point_3 ep2(to_exact(p2));
  EK::Point_3 ep3(to_exact(p3));

  return to_inexact(exact_circumcenter(ep0, ep1, ep2, ep3));
}

int main(int argc, char *argv[])
{
  const char *filename = argv[1];
  const char *outputname = (argc > 2) ? argv[2] : "voromesh.off";
  std::ifstream in(filename);

  std::string firstline;
  std::getline(in, firstline); // skip the first line
  std::stringstream ss(firstline);
  int N, M;
  ss >> N >> M;

  // load vmesh
  std::vector<std::pair<Point, int>> points;
  for (int i = 0; i < N; i++)
  {
    std::getline(in, firstline); // skip the first line
    std::stringstream ss(firstline);
    double x, y, z;
    std::cout << std::setprecision(20);
    ss >> x >> y >> z;
    points.push_back(std::make_pair(Point(x, y, z), i < M));
  }

  // create Delaunay
  DTriangulation T;
  T.insert(points.begin(), points.end());

  // Mark generators with infinite cells as outside
  std::vector<Vertex_handle> exterior_verts;
  T.adjacent_vertices(T.infinite_vertex(), std::back_inserter(exterior_verts));
  for (auto &vert : exterior_verts)
  {
    vert->info() = 0;
  }

  std::vector<PolyVertices> poly_v;
  std::vector<PolyFace> poly_f;
  int current_index = 0;

  std::map<Cell_handle, int> cell_to_int_map;

  for (Edge_iterator eit = T.finite_edges_begin(); eit != T.finite_edges_end(); ++eit)
  {
    // check if generators of different occupancies
    bool is_f = (eit->first->vertex(eit->second)->info());
    bool is_s = (eit->first->vertex(eit->third)->info());
    if (is_f != is_s)
    {
      PolyVertices duals;
      PolyFace indices;
      std::vector<Cell_handle> incident_cells;

      Cell_circulator circ = T.incident_cells(*eit);
      Cell_circulator end = circ;
      CGAL_For_all(circ, end)
      {
        if (cell_to_int_map.count(circ))
        {
          indices.push_back(cell_to_int_map.find(circ)->second);
        }
        else
        {
          Point p = exact_dual(T, circ);
          cell_to_int_map.insert(std::make_pair(circ, current_index));
          duals.push_back(p);
          indices.push_back(current_index);
          current_index++;
        }
      }
      if (is_s)
      {
        std::reverse(indices.begin(), indices.end());
      }

      poly_v.push_back(duals);

      // polygon face
      // poly_f.push_back(indices);

      // triangle face
      int fs = indices.size();
      for (int j = 2; j < fs; j++)
      {
        PolyFace triface;
        triface.push_back(indices[0]);
        triface.push_back(indices[j - 1]);
        triface.push_back(indices[j]);
        poly_f.push_back(triface);
      }
    }
  }

  std::ofstream file;
  file.open(outputname, std::ofstream::out | std::ofstream::trunc);
  file << "OFF" << std::endl;
  file << current_index << " " << poly_f.size() << " 0" << std::endl;

  for (auto &pv : poly_v)
  {
    for (auto &p : pv)
    {
      file << std::setprecision(20) << p << std::endl;
      ;
    }
  }
  for (auto &fv : poly_f)
  {
    file << "3 ";
    for (auto &f : fv)
    {
      file << f << " ";
    }
    file << std::endl;
  }
  file.close();
  return 0;
}