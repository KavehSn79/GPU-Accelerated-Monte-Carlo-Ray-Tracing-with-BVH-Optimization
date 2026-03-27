#ifndef BVH_H
#define BVH_H

#include <vector>
#include <algorithm>
#include <cstdlib>
#include "aabb.h"
#include "sphere.h"
#include "hittable.h"
#include "interval.h"

struct bvh_node {
    aabb box;
    int left;      // index of left child node, or -1
    int right;     // index of right child node, or -1
    int start;     // start index into prim_indices
    int count;     // number of primitives in leaf
    int is_leaf;   // 1 leaf, 0 internal
};

inline float rand01_host() {
    return float(std::rand()) / (float(RAND_MAX) + 1.0f);
}

inline int pick_axis() {
    return int(3.0f * rand01_host());
}

inline float sphere_centroid_axis(const sphere& s, int axis) {
    return s.center[axis];
}

// Build returns node index
inline int build_bvh_cpu(const sphere* spheres,
                         std::vector<int>& prim_indices,
                         int start, int end,
                         std::vector<bvh_node>& nodes)
{
    bvh_node node{};
    node.left = node.right = -1;
    node.start = start;
    node.count = end - start;

    // Compute bounds of this range
    aabb bounds = spheres[prim_indices[start]].bbox();
    for (int i = start + 1; i < end; i++) {
        bounds = surrounding_box(bounds, spheres[prim_indices[i]].bbox());
    }
    node.box = bounds;

    const int n = end - start;

    // Leaf threshold
    const int LEAF_SIZE = 4;
    if (n <= LEAF_SIZE) {
        node.is_leaf = 1;
        int my_index = (int)nodes.size();
        nodes.push_back(node);
        return my_index;
    }

    // Choose split axis
    int axis = pick_axis();

    // Sort primitive indices
    std::sort(prim_indices.begin() + start, prim_indices.begin() + end,
              [&](int a, int b) {
                  return sphere_centroid_axis(spheres[a], axis) <
                         sphere_centroid_axis(spheres[b], axis);
              });

    int mid = start + n / 2;

    int my_index = (int)nodes.size();
    nodes.push_back(node);

    int left_index  = build_bvh_cpu(spheres, prim_indices, start, mid, nodes);
    int right_index = build_bvh_cpu(spheres, prim_indices, mid, end, nodes);

    nodes[my_index].is_leaf = 0;
    nodes[my_index].left = left_index;
    nodes[my_index].right = right_index;
    nodes[my_index].start = -1;
    nodes[my_index].count = 0;
    nodes[my_index].box = surrounding_box(nodes[left_index].box, nodes[right_index].box);

    return my_index;
}

//GPU traversal
__device__ inline bool hit_world_bvh(const sphere* spheres,
                                     const int sphere_count,
                                     const int* prim_indices,
                                     const bvh_node* nodes,
                                     int root,
                                     const ray& r,
                                     interval ray_t,
                                     hit_record& rec)
{
    bool hit_any = false;
    float closest = ray_t.max;

    int stack[64];
    int sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
        int ni = stack[--sp];
        const bvh_node& node = nodes[ni];

        if (!node.box.hit(r, interval(ray_t.min, closest)))
            continue;

        if (node.is_leaf) {
            for (int i = 0; i < node.count; i++) {
                int prim_id = prim_indices[node.start + i];
                hit_record tmp;
                if (spheres[prim_id].hit(r, interval(ray_t.min, closest), tmp)) {
                    hit_any = true;
                    closest = tmp.t;
                    rec = tmp;
                }
            }
        } else {
            stack[sp++] = node.left;
            stack[sp++] = node.right;
        }
    }

    return hit_any;
}

#endif
