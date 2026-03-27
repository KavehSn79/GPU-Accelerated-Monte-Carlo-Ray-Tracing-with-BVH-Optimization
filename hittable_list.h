#ifndef HITTABLELIST_H
#define HITTABLELIST_H

#include "sphere.h"
#include "hittable.h"
#include "interval.h"

#define MAX_SPHERES 700

struct hittable_list {
    int count;
    sphere spheres[MAX_SPHERES];
};

__device__ bool hit_world(const hittable_list& w, const ray& r, interval ray_t, hit_record& rec) {
    hit_record temp_rec;
    bool hit_anything = false;
    float closest_so_far = ray_t.max;

    for (int i = 0; i < w.count; i++) {
        if (w.spheres[i].hit(r, interval(ray_t.min, closest_so_far), temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}

#endif
