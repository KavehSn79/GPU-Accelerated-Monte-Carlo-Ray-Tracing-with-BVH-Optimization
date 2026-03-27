#ifndef AABB_H
#define AABB_H


#include "main.h"

struct aabb {
    point3 mn;
    point3 mx;

    __host__ __device__ aabb() : mn(point3(+infinity,+infinity,+infinity)),
                                 mx(point3(-infinity,-infinity,-infinity)) {}

    __host__ __device__ aabb(const point3& a, const point3& b) : mn(a), mx(b) {}

    __host__ __device__ inline point3 min() const { return mn; }
    __host__ __device__ inline point3 max() const { return mx; }

    __host__ __device__ bool hit(const ray& r, interval t) const {
        for (int a = 0; a < 3; a++) {
            float invD = 1.0f / r.direction()[a];
            float t0 = (mn[a] - r.origin()[a]) * invD;
            float t1 = (mx[a] - r.origin()[a]) * invD;
            if (invD < 0.0f) { float tmp = t0; t0 = t1; t1 = tmp; }
            t.min = t0 > t.min ? t0 : t.min;
            t.max = t1 < t.max ? t1 : t.max;
            if (t.max <= t.min) return false;
        }
        return true;
    }
};

__host__ __device__ inline aabb surrounding_box(const aabb& box0, const aabb& box1) {
    point3 small(
        fminf(box0.min().x(), box1.min().x()),
        fminf(box0.min().y(), box1.min().y()),
        fminf(box0.min().z(), box1.min().z())
    );
    point3 big(
        fmaxf(box0.max().x(), box1.max().x()),
        fmaxf(box0.max().y(), box1.max().y()),
        fmaxf(box0.max().z(), box1.max().z())
    );
    return aabb(small, big);
}

#endif
