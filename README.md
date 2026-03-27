# Code Structure & Optimization Notes

---

## 1) Code structure (what each module does)

### Core math / utilities
- **`vec3.h`**  
  GPU-friendly 3D vector type used for geometry and colors. Implements arithmetic, dot/cross, normalization, etc.
- **`ray.h`**  
  Ray representation: `origin`, `direction`, and `at(t)` to evaluate a point along the ray.
- **`interval.h`**  
  Utility numeric range `[min,max]` used for ray `t` bounds and robust intersection tests.
- **`color.h`**  
  Converts accumulated radiance to RGB output, including gamma correction.

### Scene + intersections
- **`hittable.h`**  
  Defines `hit_record` (hit position, normal, t, material id, front-face flag).
- **`sphere.h`**  
  Sphere primitive and ray–sphere intersection logic.
- **`hittable_list.h`**  
  A fixed-size array of spheres (`MAX_SPHERES`), plus world-hit helpers.

### Acceleration structure
- **`aabb.h`**  
  Axis-aligned bounding box + ray–AABB test.
- **`bvh.h`**  
  BVH node definition and CPU-side BVH builder (`build_bvh_cpu`), plus GPU traversal hit function (`hit_world_bvh`).

### Camera + materials
- **`camera.h`**  
  Camera model and ray generation per pixel (with random sampling for anti-aliasing / DoF).
- **`material.h`**  
  Material system and scattering:
  Lambertian (diffuse), Metal, Dielectric. Uses `curandState` for random sampling.

### Entry point / CUDA kernels
- **`main.h`**  
  Common includes / constants.
- **`main.cu`**  
  Program entry point and GPU kernels:
  - `render_init(...)`
  - `render(...)`
  - `ray_color(...)`

---

## 2) Runtime pipeline (high level)

1. CPU builds the scene and BVH  
2. Scene data is copied to GPU memory  
3. GPU initializes per-pixel random states  
4. Each thread traces multiple stochastic rays  
5. Rays bounce via material scattering  
6. Radiance samples are averaged into a linear framebuffer  

---

## 3) Project Poster

![Project Poster](./poster.png)

> The poster highlights:
> - Rendering pipeline  
> - BVH acceleration  
> - Performance improvements  
> - Final visual output  

---

## 4) Optimization strategies tried

### A) Initial GPU Implementation — OOP Without BVH
Used virtual functions + O(N) intersection → slow

### B) Data-Oriented Design
≈ 2× speedup

### C) BVH Acceleration
- Runtime ≈ 6.7 s  
- Bottleneck: register-limited  

### D) Local RNG State
Avoid global memory access inside loops

### E) Coalesced Memory Access
Linear framebuffer improves memory throughput

### F) Block Size Tuning
8×8 chosen as stable configuration

### G) Removing Unified Memory
- Runtime ≈ 5.3 s  
- ~23% faster  

### H) `__noinline__`
- Registers: 84 → 66  
- Occupancy: 37% → 52%  

### I) Russian Roulette
- Runtime ≈ 3.0 s  
- Biggest runtime improvement  

---

## 5) Failed Optimizations

### Shared Memory Caching
- Occupancy dropped → slower (5.4 s)

### Larger Block Sizes
- Not consistently better → stayed with 8×8

---

## 6) Output Image

![Rendered Output](./output.png)

> Generated as `.ppm` and converted to `.png`

---

## 7) Summary

- Biggest wins: BVH + Data-oriented design + Russian roulette  
- Useful: RNG locality + memory coalescing  
- Not useful: Shared memory caching  

---

## 8) Build & Run

```bash
make clean
make
./cudart > out.ppm
گ÷
```
## 9) References

- **Ray Tracing in One Weekend**  
  https://raytracing.github.io/books/RayTracingInOneWeekend.html  

Authors: Peter Shirley, Trevor David Black, Steve Hollasch  

This project is based on the concepts from the book, extended with CUDA acceleration, BVH structures, and performance optimizations.
