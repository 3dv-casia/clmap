#pragma once

#include <unordered_map>
#include <unordered_set>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

#include "clmap/base/infinite_plane3d.h"

#include "base/linetrack.h"
#include "base/pointtrack.h"

namespace limap {

template <typename TYPE1, typename TYPE2>
class Bipartite {
 public:
  Bipartite() {}
  Bipartite(const Bipartite& bpt);
  Bipartite(py::dict dict);
  py::dict as_dict() const;

  void AddObj1(const TYPE1& obj1, int obj1_id);
  void AddObj2(const TYPE2& obj2, int obj2_id);
  void AddEdge(int obj1_id, int obj2_id, double sim);

  bool ExistObj1(int obj1_id) const;
  bool ExistObj2(int obj2_id) const;

  // note: do not clear node
  void ClearEdges();

  void UpdateObj1(int obj1_id, const TYPE1& obj1);
  void UpdateObj2(int obj2_id, const TYPE2& obj2);

  void DeleteObj1(int obj1_id);
  void DeleteObj2(int obj2_id);

  std::vector<int> GetObj1Ids() const;
  std::vector<int> GetObj2Ids() const;

  std::vector<TYPE2> GetObj1Neighbors(int obj1_id) const;
  std::vector<TYPE1> GetObj2Neighbors(int obj2_id) const;

  std::unordered_map<int, TYPE1> obj1_map;
  std::unordered_map<int, TYPE2> obj2_map;
  std::unordered_map<int, std::unordered_map<int, double>> n_1_to_2;
  std::unordered_map<int, std::unordered_map<int, double>> n_2_to_1;
};

typedef Bipartite<PointTrack, InfinitePlane3d> PP_Bipartite3d;
typedef Bipartite<LineTrack, InfinitePlane3d> LP_Bipartite3d;

////////////////////////////////////////////////////////////////////////////////
// Implement
////////////////////////////////////////////////////////////////////////////////

template <typename TYPE1, typename TYPE2>
Bipartite<TYPE1, TYPE2>::Bipartite(const Bipartite& bpt)
    : obj1_map(bpt.obj1_map),
      obj2_map(bpt.obj2_map),
      n_1_to_2(bpt.n_1_to_2),
      n_2_to_1(bpt.n_2_to_1) {}

template <typename TYPE1, typename TYPE2>
Bipartite<TYPE1, TYPE2>::Bipartite(py::dict dict) {
#define TMPMAPTYPE std::unordered_map<int, TYPE1>
  ASSIGN_PYDICT_ITEM(dict, obj1_map, TMPMAPTYPE)
#undef TMPMAPTYPE
#define TMPMAPTYPE std::unordered_map<int, TYPE2>
  ASSIGN_PYDICT_ITEM(dict, obj2_map, TMPMAPTYPE)
#undef TMPMAPTYPE
#define TMPMAPTYPE std::unordered_map<int, std::unordered_map<int, double>>
  ASSIGN_PYDICT_ITEM(dict, n_1_to_2, TMPMAPTYPE)
  ASSIGN_PYDICT_ITEM(dict, n_2_to_1, TMPMAPTYPE)
#undef TMPMAPTYPE
}

template <typename TYPE1, typename TYPE2>
py::dict Bipartite<TYPE1, TYPE2>::as_dict() const {
  py::dict output;
  output["obj1_map"] = obj1_map;
  output["obj2_map"] = obj2_map;
  output["n_1_to_2"] = n_1_to_2;
  output["n_2_to_1"] = n_2_to_1;
  return output;
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::AddObj1(const TYPE1& obj1, int obj1_id) {
  THROW_CUSTOM_CHECK_MSG(!ExistObj1(obj1_id), std::logic_error,
                         "Error: obj1_id was inserted repeatedly.");
  obj1_map.emplace(obj1_id, obj1);
  n_1_to_2.emplace(obj1_id, std::unordered_map<int, double>());
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::AddObj2(const TYPE2& obj2, int obj2_id) {
  THROW_CUSTOM_CHECK_MSG(!ExistObj2(obj2_id), std::logic_error,
                         "Error: obj2_id was inserted repeatedly.");
  obj2_map.emplace(obj2_id, obj2);
  n_2_to_1.emplace(obj2_id, std::unordered_map<int, double>());
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::AddEdge(int obj1_id, int obj2_id, double sim) {
  THROW_CUSTOM_CHECK_MSG(
      ExistObj1(obj1_id), std::logic_error,
      "Error: obj1_id doesn't exist, please insert it in advance.");
  THROW_CUSTOM_CHECK_MSG(
      ExistObj2(obj2_id), std::logic_error,
      "Error: obj2_id doesn't exist, please insert it in advance.");
  n_1_to_2.at(obj1_id).emplace(obj2_id, sim);
  n_2_to_1.at(obj2_id).emplace(obj1_id, sim);
}

template <typename TYPE1, typename TYPE2>
bool Bipartite<TYPE1, TYPE2>::ExistObj1(int obj1_id) const {
  return obj1_map.find(obj1_id) != obj1_map.end();
}

template <typename TYPE1, typename TYPE2>
bool Bipartite<TYPE1, TYPE2>::ExistObj2(int obj2_id) const {
  return obj2_map.find(obj2_id) != obj2_map.end();
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::ClearEdges() {
  for (auto it = n_1_to_2.begin(); it != n_1_to_2.end(); it++) {
    it->second.clear();
  }
  for (auto it = n_2_to_1.begin(); it != n_2_to_1.end(); it++) {
    it->second.clear();
  }
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::UpdateObj1(int obj1_id, const TYPE1& obj1) {
  THROW_CUSTOM_CHECK_MSG(
      ExistObj1(obj1_id), std::logic_error,
      "Error: obj1_id doesn't exist, please insert it in advance.");
  obj1_map.at(obj1_id) = obj1;
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::UpdateObj2(int obj2_id, const TYPE2& obj2) {
  THROW_CUSTOM_CHECK_MSG(
      ExistObj2(obj2_id), std::logic_error,
      "Error: obj2_id doesn't exist, please insert it in advance.");
  obj2_map.at(obj2_id) = obj2;
}

template <typename TYPE1, typename TYPE2>
std::vector<TYPE2> Bipartite<TYPE1, TYPE2>::GetObj1Neighbors(
    int obj1_id) const {
  THROW_CUSTOM_CHECK_MSG(
      ExistObj1(obj1_id), std::logic_error,
      "Error: obj1_id doesn't exist, please insert it in advance.");
  std::vector<TYPE2> neighbors_vec;
  const auto& neighbors = n_1_to_2.at(obj1_id);
  for (const auto& neighbor : neighbors) {
    int obj2_id = neighbor.first;
    neighbors_vec.push_back(obj2_map.at(obj2_id));
  }
  return neighbors_vec;
}

template <typename TYPE1, typename TYPE2>
std::vector<TYPE1> Bipartite<TYPE1, TYPE2>::GetObj2Neighbors(
    int obj2_id) const {
  THROW_CUSTOM_CHECK_MSG(
      ExistObj2(obj2_id), std::logic_error,
      "Error: obj2_id doesn't exist, please insert it in advance.");
  std::vector<TYPE1> neighbors_vec;
  const auto& neighbors = n_2_to_1.at(obj2_id);
  for (const auto& neighbor : neighbors) {
    int obj1_id = neighbor.first;
    neighbors_vec.push_back(obj1_map.at(obj1_id));
  }
  return neighbors_vec;
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::DeleteObj1(int obj1_id) {
  THROW_CUSTOM_CHECK_MSG(
      ExistObj1(obj1_id), std::logic_error,
      "Error: obj1_id doesn't exist, please insert it in advance.");
  const auto& neighbors = n_1_to_2.at(obj1_id);
  for (const auto& neighbor : neighbors) {
    int obj2_id = neighbor.first;
    n_2_to_1.at(obj2_id).erase(obj1_id);
  }
  obj1_map.erase(obj1_id);
  n_1_to_2.erase(obj1_id);
}

template <typename TYPE1, typename TYPE2>
void Bipartite<TYPE1, TYPE2>::DeleteObj2(int obj2_id) {
  THROW_CUSTOM_CHECK_MSG(
      ExistObj2(obj2_id), std::logic_error,
      "Error: obj2_id doesn't exist, please insert it in advance.");
  const auto& neighbors = n_2_to_1.at(obj2_id);
  for (const auto& neighbor : neighbors) {
    int obj1_id = neighbor.first;
    n_1_to_2.at(obj1_id).erase(obj2_id);
  }
  obj2_map.erase(obj2_id);
  n_2_to_1.erase(obj2_id);
}

template <typename TYPE1, typename TYPE2>
std::vector<int> Bipartite<TYPE1, TYPE2>::GetObj1Ids() const {
  std::vector<int> obj1_ids;
  obj1_ids.reserve(obj1_map.size());
  for (auto it = obj1_map.begin(); it != obj1_map.end(); it++) {
    obj1_ids.push_back(it->first);
  }
  return obj1_ids;
}

template <typename TYPE1, typename TYPE2>
std::vector<int> Bipartite<TYPE1, TYPE2>::GetObj2Ids() const {
  std::vector<int> obj2_ids;
  obj2_ids.reserve(obj2_map.size());
  for (auto it = obj2_map.begin(); it != obj2_map.end(); it++) {
    obj2_ids.push_back(it->first);
  }
  return obj2_ids;
}

}  // namespace limap
