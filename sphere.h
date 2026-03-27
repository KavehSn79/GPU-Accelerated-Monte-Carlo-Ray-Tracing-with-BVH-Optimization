#ifndef SPHERE_H
#define SPHERE_H

#include "vec3.h"
#include "ray.h"
#include "interval.h"
#include "hittable.h"
#include "aabb.h"

struct sphere {
    point3 center;
    float radius;
    int mat_id;

    __host__ __device__ sphere() : center(), radius(0), mat_id(0) {}

    __host__ __device__ sphere(const point3& c, float r, int m) : center(c), radius(r), mat_id(m) {}

    __device__ bool hit(const ray& r, interval ray_t, hit_record& rec) const {
        vec3 oc = center - r.origin();
        float a = r.direction().length_squared();
        float h = dot(r.direction(), oc);
        float c = oc.length_squared() - radius*radius;
        float discriminant = h*h - a*c;

        if (discriminant < 0.0f) return false;

        float sqrtd = sqrtf(discriminant);

        float root = (h - sqrtd) / a;
        if (!ray_t.surrounds(root)) {
            root = (h + sqrtd) / a;
            if (!ray_t.surrounds(root)) return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        rec.mat_id = mat_id;            

        return true;
    }
    __host__ __device__ inline aabb bbox() const {
        vec3 r(radius, radius, radius);
        return aabb(center - r, center + r);
}
};





#endif
