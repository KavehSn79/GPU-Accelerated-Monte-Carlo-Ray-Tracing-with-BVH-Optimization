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
  - `render_init(...)`: initialize CURAND states per pixel
  - `render(...)`: path trace samples per pixel, store in linear framebuffer
  - `ray_color(...)`: iterative bounce loop (up to 20), uses BVH traversal + material scatter

---

## 2) Runtime pipeline (high level)

1. CPU builds the scene and BVH  
2. Scene data is copied to GPU memory  
3. GPU initializes per-pixel random states  
4. Each thread traces multiple stochastic rays  
5. Rays bounce via material scattering  
6. Radiance samples are averaged into a linear framebuffer.

The framebuffer is a **1D linear array** (`screen[pixel_index] = ty*max_x + tx`).

---

## 3) Optimization strategies tried

### A) Initial GPU Implementation — OOP Without BVH

Used **virtual functions and polymorphic objects** and **O(N)** intersection.  
Caused dynamic dispatch overhead, poor locality, and high intersection cost.

**Main bottlenecks:**  
OOP abstraction overhead + linear intersection complexity.

---

### B) Data-Oriented Design (Still Without BVH)

Replaced polymorphism with **plain structs, direct calls, and contiguous memory**.

**Result:**  
≈ **2× speedup** before BVH.  
Largest **structural optimization**.

---

### C) BVH Acceleration

Reduced intersection complexity to **≈ O(log N)**.

**Nsight baseline after BVH:**
- Runtime ≈ **6.7 s**
- Registers ≈ **84**
- Occupancy ≈ **37%**
- Compute ≈ **62%**, Memory ≈ **62%**
- Bottleneck: **register-limited**

---

### D) Per-thread RNG state kept in registers (local copy)
**What:** Load RNG state to a local variable, use it for all samples, then write it back once.  
**Where:** `render(...)` in `main.cu`:
- `curandState local_rand_state = rand_state[pixel_index];`
- Use `local_rand_state` in the sample loop
- `rand_state[pixel_index] = local_rand_state;`

**Why it helps:** Avoids repeated global memory reads/writes of CURAND state inside the inner sampling loop.

---

### E) Coalesced framebuffer writes
**What:** Use a linear framebuffer and map threads so adjacent threads write adjacent pixels.  
**Where:** `render(...)` in `main.cu`:
- `pixel_index = ty*max_x + tx`
- `screen[pixel_index] = ...`

**Why it helps:** Writes are naturally **coalesced** for threads that differ in `tx` within a warp.

---

### F) Kernel configuration exploration (block size)
**What:** Tested different block sizes; current runs use **8×8** blocks.  
**Why it matters:** Block size affects occupancy, register pressure, and memory access behavior.  
**Outcome:** 8×8 was a stable choice for my setup (and matched good profiler behavior).

---

### G) Removing Unified Memory

Switched to **explicit device allocations**.

- Runtime ≈ **5.3 s** (**23% faster**)  
- Throughput balanced (~63/63)  
- Still register-limited  

---

### H) `__noinline__` to Reduce Register Pressure

- Registers **84 → 66**  
- Occupancy **37% → 52%**  
- Runtime ≈ **4.4 s** (**16% faster**)  
- Throughput ≈ **76%**

---

### I) Russian Roulette Termination

- Runtime ≈ **3.0 s** (**32% faster**)  
- Registers ≈ **70**  
- Throughput still ≈ **76%**  

Major gain from **shorter average ray paths**.

---

## 4) “Failed”

### 1) Shared Memory Scene Caching

- Shared memory ≈ **21 KB/block**
- Occupancy dropped to **≈16%**
- Kernel became latency-bound
- Runtime **3.0 s → 5.4 s (worse)**


### 2) Changing block size beyond the chosen configuration
**Idea:** Larger blocks (e.g., 16×16) to increase occupancy and memory coalescing.  
**What happened:** Not consistently better; depending on registers and divergence, it could reduce occupancy or increase pressure, offsetting theoretical gains.  
**Outcome:** Stayed with a block size that remained stable under profiling -> 8×8

---

## 5) Summary of what matters most

- **Biggest speedups:** BVH acceleration + avoiding OOP/virtual dispatch on GPU + Russian roulette.  
- **Helpful micro-optimizations:** Local RNG state, linear framebuffer, sensible block size.  
- **Not worth it here:** Shared memory for BVH/spheres (low reuse + divergent traversal).

---

## 6) How to Build and Render:
- Open project folder in terminal
- make clean
- make
- ./cudart > out.ppm
