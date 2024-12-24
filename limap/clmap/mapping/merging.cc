#include "clmap/mapping/merging.h"

#include "clmap/mapping/line_triangulation.h"

#include "base/graph.h"
#include "merging/aggregator.h"

namespace limap {

std::vector<int> ComputeLineTrackLabelsCollinearity2D(
    const Graph& graph, const std::vector<Line3d>& line3d_list_nodes,
    const std::unordered_map<image_t, std::vector<Line2d>>& all_line2D,
    const double& overlap_th, const double& angle_th,
    const double& perp_dist_th) {
  const size_t n_nodes = graph.nodes.size();
  std::vector<edge_tuple> edges;

  for (const Edge* edge_ptr : graph.undirected_edges) {
    double edge_score =
        edge_ptr->sim;  // [0.5, 1.0]  1.0 may come from overlap score
    THROW_CHECK_GE(edge_score, 0.5)
    THROW_CHECK_LE(edge_score, 1.0)
    size_t node_idx1 = edge_ptr->node_idx1;
    size_t node_idx2 = edge_ptr->node_idx2;
    edges.emplace_back(edge_score, node_idx1, node_idx2);
  }

  std::sort(edges.begin(), edges.end(),
            [](const edge_tuple& edge1, const edge_tuple& edge2) {
              return std::get<0>(edge1) > std::get<0>(edge2);
            });

  std::cout << "# graph nodes: " << n_nodes << std::endl;
  std::cout << "# graph edges: " << edges.size() * 2 << std::endl;

  std::vector<int> parent_nodes(n_nodes, -1);

  std::vector<std::set<int>> images_in_track(n_nodes);
  std::vector<std::set<int>> nodes_in_track(n_nodes);
  std::vector<std::vector<Line3d>> lines_in_track(n_nodes);

  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    // self
    images_in_track[node_idx].insert(graph.nodes[node_idx]->image_idx);
    nodes_in_track[node_idx].insert(node_idx);
    lines_in_track[node_idx].push_back(line3d_list_nodes[node_idx]);
  }

  size_t n_edges = edges.size();
  progressbar bar(n_edges);
  for (size_t edge_id = 0; edge_id < n_edges; ++edge_id) {
    bar.update();
    const auto& e = edges[edge_id];
    size_t node_idx1 = std::get<1>(e);
    size_t node_idx2 = std::get<2>(e);

    size_t root1 = limap::union_find_get_root(node_idx1, parent_nodes);
    size_t root2 = limap::union_find_get_root(node_idx2, parent_nodes);

    if (root1 != root2) {
      bool flag = true;
      for (auto it1 = nodes_in_track[root1].begin();
           it1 != nodes_in_track[root1].end(); ++it1) {
        const PatchNode* node1 = graph.nodes[*it1];
        image_t image_id1 = node1->image_idx;
        line2d_t line2d_idx1 = node1->line_idx;
        const Line2d& line2d1 = all_line2D.at(image_id1)[line2d_idx1];

        for (auto it2 = nodes_in_track[root2].begin();
             it2 != nodes_in_track[root2].end(); ++it2) {
          const PatchNode* node2 = graph.nodes[*it2];
          image_t image_id2 = node2->image_idx;
          line2d_t line2d_idx2 = node2->line_idx;
          const Line2d& line2d2 = all_line2D.at(image_id2)[line2d_idx2];
          if (image_id1 == image_id2) {
            if (TestDifferentInfLines3D(line2d1, line2d2, overlap_th, angle_th,
                                        perp_dist_th)) {
              // cannot link
              flag = false;
              break;
            }
          }
        }
        if (!flag) break;
      }
      if (!flag) continue;

      // Union-find merging heuristic
      // Weight balanced Union
      if (images_in_track[root1].size() < images_in_track[root2].size()) {
        parent_nodes[root1] = root2;
        // update images_in_track for root2
        images_in_track[root2].insert(images_in_track[root1].begin(),
                                      images_in_track[root1].end());
        images_in_track[root1].clear();
        // update nodes_in_track for root2
        nodes_in_track[root2].insert(nodes_in_track[root1].begin(),
                                     nodes_in_track[root1].end());
        nodes_in_track[root1].clear();
        // update lines_in_track for root2
        lines_in_track[root2].insert(lines_in_track[root2].end(),
                                     lines_in_track[root1].begin(),
                                     lines_in_track[root1].end());
        lines_in_track[root1].clear();
      } else {
        parent_nodes[root2] = root1;
        // update images_in_track for root1
        images_in_track[root1].insert(images_in_track[root2].begin(),
                                      images_in_track[root2].end());
        images_in_track[root2].clear();
        // update nodes_in_track for root1
        nodes_in_track[root1].insert(nodes_in_track[root2].begin(),
                                     nodes_in_track[root2].end());
        nodes_in_track[root2].clear();
        // update lines_in_track for root1
        lines_in_track[root1].insert(lines_in_track[root1].end(),
                                     lines_in_track[root2].begin(),
                                     lines_in_track[root2].end());
        lines_in_track[root2].clear();
      }
    }
  }

  // compute the track_id for each node
  std::vector<int> track_labels(n_nodes, -1);

  // only save tracks with at least two nodes
  size_t n_tracks = 0;
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1) continue;  // node_idx is parent
    size_t parent_idx = parent_nodes[node_idx];
    if (parent_nodes[parent_idx] == -1 && track_labels[parent_idx] == -1) {
      // only save root's track_id
      track_labels[parent_idx] = n_tracks++;
    }
  }
  std::cout << "# tracks: " << n_tracks << std::endl;

  // assign track_id to all node_idx
  for (size_t node_idx = 0; node_idx < n_nodes; ++node_idx) {
    if (parent_nodes[node_idx] == -1) continue;  // node_idx is parent
    track_labels[node_idx] =
        track_labels[limap::union_find_get_root(node_idx, parent_nodes)];
  }
  return track_labels;
}


std::vector<LineTrack> RemergeLineTracks(
    const std::vector<LineTrack>& linetracks, LineLinker3d linker3d,
    const int num_outliers, bool use_collinearity2D, const double& overlap_th,
    const double& angle_th, const double& perp_dist_th) {
  linker3d.config.set_to_spatial_merging();
  size_t n_tracks = linetracks.size();

  // compute edges to remerge
  std::set<std::pair<size_t, size_t>> edges;
  std::vector<std::set<std::pair<size_t, size_t>>> edges_per_track(n_tracks);
  std::vector<int> active_ids;
  for (size_t i = 0; i < n_tracks; ++i) {
    if (linetracks[i].active) active_ids.push_back(i);
  }
  int n_active_ids = active_ids.size();
  progressbar bar(n_active_ids, n_active_ids >= 10000);
#pragma omp parallel for
  for (size_t k = 0; k < n_active_ids; ++k) {
    int i = active_ids[k];
    bar.update();
    const Line3d& l1 = linetracks[i].line;
    for (size_t j = 0; j < n_tracks; ++j) {
      if (i == j) continue;
      if (n_active_ids == n_tracks) {
        if (i < j && (i + j) % 2 == 0) continue;
        if (i > j && (i + j) % 2 == 1) continue;
      }
      const Line3d& l2 = linetracks[j].line;
      bool valid = linker3d.check_connection(l1, l2);
      if (!valid) continue;
      if (i < j)
        edges_per_track[i].insert(std::make_pair(i, j));
      else
        edges_per_track[i].insert(std::make_pair(j, i));
    }
  }
  for (size_t i = 0; i < n_tracks; ++i) {
    edges.insert(edges_per_track[i].begin(), edges_per_track[i].end());
  }

  // group connected components
  std::vector<int> parent_tracks(n_tracks, -1);
  std::vector<std::set<int>> tracks_in_group(n_tracks);
  for (size_t i = 0; i < n_tracks; ++i) {
    tracks_in_group[i].insert(i);  // `i` is the track id
  }
  for (auto it = edges.begin(); it != edges.end(); ++it) {
    size_t track_id1 = it->first;
    size_t track_id2 = it->second;

    size_t root1 = union_find_get_root(track_id1, parent_tracks);
    size_t root2 = union_find_get_root(track_id2, parent_tracks);
    if (root1 != root2) {
      if (use_collinearity2D) {
        bool flag = true;
        for (const auto& track_id1 : tracks_in_group[root1]) {
          const auto& track1 = linetracks[track_id1];
          for (const auto& track_id2 : tracks_in_group[root2]) {
            const auto& track2 = linetracks[track_id2];
            for (size_t i = 0; i < track1.image_id_list.size(); ++i) {
              image_t image_id1 = track1.image_id_list[i];
              const Line2d& line2d1 = track1.line2d_list[i];
              for (size_t j = 0; j < track2.image_id_list.size(); ++j) {
                image_t image_id2 = track2.image_id_list[j];
                if (image_id1 == image_id2) {
                  const Line2d& line2d2 = track2.line2d_list[j];
                  if (TestDifferentInfLines3D(line2d1, line2d2, overlap_th,
                                              angle_th, perp_dist_th)) {
                    // cannot link
                    flag = false;
                    break;
                  }
                }
              }
              if (!flag) break;
            }
            if (!flag) break;
          }
          if (!flag) break;
        }
        if (!flag) continue;
      }

      // Union-find merging heuristic.
      if (tracks_in_group[root1].size() < tracks_in_group[root2].size()) {
        parent_tracks[root1] = root2;
        // update tracks_in_group for root2
        tracks_in_group[root2].insert(tracks_in_group[root1].begin(),
                                      tracks_in_group[root1].end());
        tracks_in_group[root1].clear();
      } else {
        parent_tracks[root2] = root1;
        // update tracks_in_group for root1
        tracks_in_group[root1].insert(tracks_in_group[root2].begin(),
                                      tracks_in_group[root2].end());
        tracks_in_group[root2].clear();
      }
    }
  }

  // compute groups
  std::vector<int> group_labels(n_tracks, -1);
  size_t n_groups = 0;
  for (size_t track_idx = 0; track_idx < n_tracks; ++track_idx) {
    if (parent_tracks[track_idx] == -1) {
      group_labels[track_idx] = n_groups++;
    }
  }
  // STDLOG(INFO) << "# groups after remerging:" << " " << n_groups <<
  // std::endl;
  for (size_t track_idx = 0; track_idx < n_tracks; ++track_idx) {
    if (group_labels[track_idx] != -1) {
      continue;
    }
    group_labels[track_idx] =
        group_labels[union_find_get_root(track_idx, parent_tracks)];
  }

  // recompute track information for each group
  std::vector<LineTrack> new_linetracks(n_groups);
  std::vector<int> counter_groups(n_groups, 0);
  for (size_t track_id = 0; track_id < n_tracks; ++track_id) {
    const LineTrack& track = linetracks[track_id];
    size_t group_id = group_labels[track_id];
    counter_groups[group_id]++;

    new_linetracks[group_id].node_id_list.insert(
        new_linetracks[group_id].node_id_list.end(), track.node_id_list.begin(),
        track.node_id_list.end());
    new_linetracks[group_id].image_id_list.insert(
        new_linetracks[group_id].image_id_list.end(),
        track.image_id_list.begin(), track.image_id_list.end());
    new_linetracks[group_id].line_id_list.insert(
        new_linetracks[group_id].line_id_list.end(), track.line_id_list.begin(),
        track.line_id_list.end());
    new_linetracks[group_id].line2d_list.insert(
        new_linetracks[group_id].line2d_list.end(), track.line2d_list.begin(),
        track.line2d_list.end());
    new_linetracks[group_id].line3d_list.insert(
        new_linetracks[group_id].line3d_list.end(), track.line3d_list.begin(),
        track.line3d_list.end());
    new_linetracks[group_id].score_list.insert(
        new_linetracks[group_id].score_list.end(), track.score_list.begin(),
        track.score_list.end());
  }

  // aggregate 3d lines to get final line proposal
  for (size_t group_id = 0; group_id < n_groups; ++group_id) {
    new_linetracks[group_id].line = merging::Aggregator::aggregate_line3d_list(
        new_linetracks[group_id].line3d_list,
        new_linetracks[group_id].score_list, num_outliers);
    if (counter_groups[group_id] == 1) {
      new_linetracks[group_id].active = false;
    }
  }
  return new_linetracks;
}

}  // namespace limap
