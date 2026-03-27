#ifndef MATERIAL_H
#define MATERIAL_H

#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "color.h"

enum material_type : int { MAT_LAMBERTIAN = 0, MAT_METAL = 1,  MAT_DIELECTRIC = 2};

struct material {
    material_type type;
    color albedo; // lambertian/metal
    float fuzz;   // only for metal
    float  refraction_index;    // dielectric refraction index
};

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v, n) * n;
}

__device__ vec3 random_unit_vector(curandState* state);
__device__ vec3 random_in_unit_sphere(curandState* state);

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    float cos_theta = fminf(dot(-uv, n), 1.0f);

    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);

    vec3 r_out_parallel =
        -sqrtf(fabsf(1.0f - r_out_perp.length_squared())) * n;

    return r_out_perp + r_out_parallel;
}

__device__ static float reflectance(float cosine, float refraction_index) {
    auto r0 = (1 - refraction_index) / (1 + refraction_index);
    r0 = r0*r0;
    return r0 + (1.0f - r0) * powf(1.0f - cosine, 5.0f);
}

__device__ inline bool scatter(const material& m,
                               const ray& r_in,
                               const hit_record& rec,
                               color& attenuation,
                               ray& scattered,
                               curandState* state)
{
    if (m.type == MAT_LAMBERTIAN) {
        vec3 scatter_dir = rec.normal + random_unit_vector(state);
        if (scatter_dir.length_squared() < 1e-12f) scatter_dir = rec.normal;
        scattered = ray(rec.p, scatter_dir);
        attenuation = m.albedo;
        return true;
    }

    if (m.type == MAT_METAL) {
        vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
        reflected = unit_vector(reflected) + (m.fuzz * random_unit_vector(state));
        scattered = ray(rec.p, reflected + m.fuzz * random_in_unit_sphere(state));
        attenuation = m.albedo;
        return dot(scattered.direction(), rec.normal) > 0.0f;
    }

    if (m.type == MAT_DIELECTRIC) {
        attenuation = color(1.0, 1.0, 1.0);
        float ri = rec.front_face ? (1.0/m.refraction_index) : m.refraction_index;
        vec3 unit_direction = unit_vector(r_in.direction());

        float cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
        float sin_theta = sqrtf(1.0 - cos_theta*cos_theta);

        bool cannot_refract = ri * sin_theta > 1.0;

        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, ri) > curand_uniform(state))
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, ri);
        
        scattered = ray(rec.p, direction);


        return true;
    }

    return false;
}

#endif
