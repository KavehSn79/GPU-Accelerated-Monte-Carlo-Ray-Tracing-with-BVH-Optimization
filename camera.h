#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"


__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * 0.01745329251994329577f; // pi/180
}

__device__ vec3 random_in_unit_disk(curandState* state) {
    while (true) {
        vec3 p(
            -1.0f + 2.0f * curand_uniform(state),
            -1.0f + 2.0f * curand_uniform(state),
            0.0f
        );
        if (p.length_squared() < 1.0f) return p;
    }
}

struct camera {
    float aspect_ratio = 16.0f / 9.0f;
    int   image_width  = 1200;
    int   image_height;

    point3 origin;
    point3 pixel00_loc;
    vec3   pixel_delta_u;
    vec3   pixel_delta_v;
    vec3   u, v, w;
    vec3   defocus_disk_u;
    vec3   defocus_disk_v;

    float vfov = 20.0f;
    point3 lookfrom = point3(13,2,3);
    point3 lookat   = point3(0,0,0);
    vec3   vup      = vec3(0,1,0);

    float defocus_angle = 0.6f;
    float focus_dist = 10.0f;

    __host__ camera() {
        image_height = int(image_width / aspect_ratio);
        image_height = (image_height < 1) ? 1 : image_height;

        origin = lookfrom;

        float theta = degrees_to_radians(vfov);
        float h = tanf(theta * 0.5f);
        float viewport_height = 2.0f * h * focus_dist;
        float viewport_width  = viewport_height * (float(image_width) / image_height);

        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        vec3 viewport_u = viewport_width * u;
        vec3 viewport_v = -viewport_height * v;

        pixel_delta_u = viewport_u / float(image_width);
        pixel_delta_v = viewport_v / float(image_height);

        point3 viewport_upper_left = origin - (focus_dist * w) - viewport_u*0.5f - viewport_v*0.5f;
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);

        float defocus_radius = focus_dist * tanf(degrees_to_radians(defocus_angle * 0.5f));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;
    }

    __device__ point3 defocus_disk_sample(curandState* state) const {
        vec3 p = random_in_unit_disk(state);
        return origin + (p.x() * defocus_disk_u) + (p.y() * defocus_disk_v);
    }

    __device__ ray get_ray(float i, float j, curandState* state) const {
        point3 ray_origin = (defocus_angle <= 0.0f) ? origin : defocus_disk_sample(state);
        vec3 ray_dir = pixel00_loc + i*pixel_delta_u + j*pixel_delta_v - ray_origin;
        return ray(ray_origin, ray_dir);
    }
};


#endif
