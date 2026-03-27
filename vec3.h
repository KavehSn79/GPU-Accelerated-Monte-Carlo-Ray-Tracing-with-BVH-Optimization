#ifndef VEC3_H
#define VEC3_H

#include <cmath>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

class vec3 {
public:
    float e[3];

    __host__ __device__ vec3() : e{0.0, 0.0, 0.0} {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{e0, e1, e2} {}

    __host__ __device__ float x() const { return e[0]; }
    __host__ __device__ float y() const { return e[1]; }
    __host__ __device__ float z() const { return e[2]; }

    __host__ __device__ vec3 operator-() const {
        return vec3(-e[0], -e[1], -e[2]);
    }

    __host__ __device__ float operator[](int i) const { return e[i]; }
    __host__ __device__ float& operator[](int i) { return e[i]; }

    __host__ __device__ vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ vec3& operator/=(float t) {
        return *this *= 1.0 / t;
    }

    __host__ __device__ float length_squared() const {
        return e[0]*e[0] + e[1]*e[1] + e[2]*e[2];
    }

    __host__ __device__ float length() const {
        return sqrtf(length_squared());
    }

    __device__ static vec3 random(curandState* state) {
        return vec3(curand_uniform(state),
                    curand_uniform(state),
                    curand_uniform(state));
    }

    //GPU: [min,max)
    __device__ static vec3 random(float min, float max, curandState* state) {
        float r0 = min + (max - min) * curand_uniform(state);
        float r1 = min + (max - min) * curand_uniform(state);
        float r2 = min + (max - min) * curand_uniform(state);
        return vec3(r0, r1, r2);
    }
};

using point3 = vec3;


__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return (1.0 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
         + u.e[1] * v.e[1]
         + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(
        u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]
    );
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ vec3 random_in_unit_sphere(curandState *state) {
    while (true) {
        vec3 p = vec3(
            curand_uniform(state),
            curand_uniform(state),
            curand_uniform(state)
        ) * 2.0f - vec3(1,1,1);

        if (p.length_squared() < 1.0f)
            return p;
    }
}

__device__ vec3 random_unit_vector(curandState* state) {
    while (true) {
        vec3 p = vec3::random(-1.0f, 1.0f, state);
        float lensq = p.length_squared();
        if (lensq > 1e-12f && lensq <= 1.0f)
            return p / sqrtf(lensq);
    }
}

__device__ vec3 random_on_hemisphere(const vec3& normal, curandState* state) {
    vec3 on_unit_sphere = random_unit_vector(state);
    if (dot(on_unit_sphere, normal) > 0.0)
        return on_unit_sphere;
    else
        return -on_unit_sphere; 
}




#endif
