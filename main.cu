#include <curand.h>
#include <curand_kernel.h>

#include "main.h"
#include "sphere.h"
#include "hittable_list.h"
#include "hittable.h"
#include "camera.h"
#include "material.h"
#include "bvh.h"




#define CHECK_CUDA(call)                                        \
    if ((call) != cudaSuccess)                                  \
    {                                                           \
        std::cerr << "CUDA error at " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE);                                     \
    }



__device__ __noinline__ vec3 ray_color(const ray& r,
                          const hittable_list* world,
                          const material* materials,
                          const int* prim_indices,
                          const bvh_node* nodes,
                          int root,
                          curandState* local_rand_state)

{
    ray cur_ray = r;
    color cur_attenuation(1.0f, 1.0f, 1.0f);

    for (int i = 0; i < 20; i++) {
        hit_record rec;

        if (hit_world_bvh(world->spheres, world->count, prim_indices, nodes, root,
            cur_ray, interval(0.001f, infinity), rec)) {
            ray scattered;
            color attenuation;

            const material& m = materials[rec.mat_id];

            if (!scatter(m, cur_ray, rec, attenuation, scattered, local_rand_state))
                return color(0,0,0);

            cur_attenuation = cur_attenuation * attenuation;
            cur_ray = scattered;

        // Russian Roulette after 5 bounces
        if (i >= 5) {
            float p = fmaxf(cur_attenuation.x(), fmaxf(cur_attenuation.y(), cur_attenuation.z()));
            p = fminf(p, 0.95f);

            if (curand_uniform(local_rand_state) > p)
                return color(0,0,0);

            cur_attenuation /= p;
        }
        } else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            color sky = (1.0f - t) * color(1.0f, 1.0f, 1.0f)
                      + t * color(0.5f, 0.7f, 1.0f);
            return cur_attenuation * sky;
        }
    }

    return color(0,0,0);
}




__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(vec3* screen, int max_x, int max_y, int number_of_samples,
                    curandState *rand_state, hittable_list *w, camera *cam, material *mats,
                    int* prim_indices, bvh_node* nodes, int* root) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    if ((tx >= max_x) || (ty >= max_y)) return;

    int pixel_index = ty * max_x + tx;
    curandState local_rand_state = rand_state[pixel_index];
    int root_val = *root;
    vec3 col(0,0,0);
    for (int sample = 0; sample < number_of_samples; sample++) {
        float u = float(tx) + curand_uniform(&local_rand_state);
        float v = float(ty) + curand_uniform(&local_rand_state);
        ray r = cam->get_ray(u, v, &local_rand_state);
        col += ray_color(r, w, mats, prim_indices, nodes, root_val, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;
    screen[pixel_index] = col / float(number_of_samples);
}


int main() {

    // Setup world

    hittable_list w;
    w.count = 0;

    const int MAX_MATS = 600;

    material* h_mats = (material*)malloc(MAX_MATS * sizeof(material));
    if (!h_mats) { std::cerr << "malloc failed\n"; return 1; }

    material* d_mats = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_mats, MAX_MATS * sizeof(material)));

    int mat_count = 0;

    auto rnd01 = []() -> float {
        return float(std::rand()) / (float(RAND_MAX) + 1.0f); // [0,1)
    };
    auto rnd = [&](float minv, float maxv) -> float {
        return minv + (maxv - minv) * rnd01();
    };
    auto random_color = [&](float minv, float maxv) -> color {
        return color(rnd(minv, maxv), rnd(minv, maxv), rnd(minv, maxv));
    };

    // ground material
    {
        int ground_id = mat_count++;
        h_mats[ground_id] = { MAT_LAMBERTIAN, color(0.5f, 0.5f, 0.5f), 0.0f, 0.0f };

        w.spheres[w.count++] = sphere(point3(0.0f, -1000.0f, 0.0f), 1000.0f, ground_id);
    }

    // random small spheres
    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            float choose_mat = rnd01();

            point3 center(
                float(a) + 0.9f * rnd01(),
                0.2f,
                float(b) + 0.9f * rnd01()
            );

            if ((center - point3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {

                int id = mat_count++;

                if (choose_mat < 0.8f) {
                    // diffuse
                    color albedo = random_color(0.0f, 1.0f) * random_color(0.0f, 1.0f);
                    h_mats[id] = { MAT_LAMBERTIAN, albedo, 0.0f, 0.0f };
                }
                else if (choose_mat < 0.95f) {
                    // metal
                    color albedo = random_color(0.5f, 1.0f);
                    float fuzz = rnd(0.0f, 0.5f);
                    h_mats[id] = { MAT_METAL, albedo, fuzz, 0.0f };
                }
                else {
                    // glass
                    h_mats[id] = { MAT_DIELECTRIC, color(1.0f, 1.0f, 1.0f), 0.0f, 1.5f };
                }

                w.spheres[w.count++] = sphere(center, 0.2f, id);
            }
        }
    }

    // three big spheres
    {
        int id1 = mat_count++;
        h_mats[id1] = { MAT_DIELECTRIC, color(1.0f, 1.0f, 1.0f), 0.0f, 1.5f };
        w.spheres[w.count++] = sphere(point3(0.0f, 1.0f, 0.0f), 1.0f, id1);

        int id2 = mat_count++;
        h_mats[id2] = { MAT_LAMBERTIAN, color(0.4f, 0.2f, 0.1f), 0.0f, 0.0f };
        w.spheres[w.count++] = sphere(point3(-4.0f, 1.0f, 0.0f), 1.0f, id2);

        int id3 = mat_count++;
        h_mats[id3] = { MAT_METAL, color(0.7f, 0.6f, 0.5f), 0.0f, 0.0f };
        w.spheres[w.count++] = sphere(point3(4.0f, 1.0f, 0.0f), 1.0f, id3);
    }

    // copy material to gpu
    CHECK_CUDA(cudaMemcpy(d_mats, h_mats, mat_count * sizeof(material), cudaMemcpyHostToDevice));
    
    //BVH

    std::vector<int> prim_indices(w.count);
    for (int i = 0; i < w.count; i++) prim_indices[i] = i;

    std::vector<bvh_node> bvh_nodes;
    bvh_nodes.reserve(2 * w.count);

    int root = build_bvh_cpu(w.spheres, prim_indices, 0, w.count, bvh_nodes);

    // Copy BVH to GPU
    bvh_node* d_nodes = nullptr;
    int* d_prim_indices = nullptr;

    CHECK_CUDA(cudaMalloc((void **)&d_nodes, bvh_nodes.size() * sizeof(bvh_node)));
    CHECK_CUDA(cudaMalloc((void **)&d_prim_indices, prim_indices.size() * sizeof(int)));

    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(d_nodes,bvh_nodes.data(),
                      bvh_nodes.size() * sizeof(bvh_node),
                      cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaMemcpy(d_prim_indices, prim_indices.data(), 
            prim_indices.size() * sizeof(int), cudaMemcpyHostToDevice));



    int h_root = root;
    int h_node_count = (int)bvh_nodes.size();


    int* d_root = nullptr;
    int* d_node_count = nullptr;
    
    CHECK_CUDA(cudaMalloc((void **)&d_root, sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_node_count, sizeof(int)));

    
    CHECK_CUDA(cudaMemcpy(d_root, &h_root, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_node_count, &h_node_count, sizeof(int), cudaMemcpyHostToDevice));



    // Camera and Image initialization
    camera cam = camera();
    std::cerr << "Spheres built: " << w.count << "\n";
    std::cerr << "Materials built: " << mat_count << "\n";

    // GPU framebuffer
    int image_width = cam.image_width;
    int image_height = cam.image_height;
    int numPixels = image_width * image_height;
    size_t screen_size = numPixels * sizeof(vec3);
    vec3* d_screen = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_screen, screen_size));

    vec3* h_screen = (vec3*)malloc(screen_size);
    if (!h_screen) { std::cerr << "malloc failed\n"; return 1; }



    // curand
    curandState *d_rand_state;
    CHECK_CUDA(cudaMalloc((void **)&d_rand_state, numPixels*sizeof(curandState)));


    // Render
    int tx = 8;
    int ty = 8;
    int number_of_samples = 500;
    dim3 blocks(image_width/tx+1, image_height/ty+1);
    dim3 threads(tx, ty);
    std::cerr << "Rendering a " << image_width << "x" << image_height << " image ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";
    
    //Alloc
    hittable_list* d_world = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_world, sizeof(hittable_list)));

    // copy host struct to device
    CHECK_CUDA(cudaMemcpy(d_world, &w, sizeof(hittable_list), cudaMemcpyHostToDevice));

    camera* d_cam = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_cam, sizeof(camera)));
    CHECK_CUDA(cudaMemcpy(d_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice));


    clock_t start, stop;
    start = clock();
    render_init<<<blocks, threads>>>(image_width, image_height, d_rand_state);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    render<<<blocks, threads>>>(d_screen, image_width, image_height, number_of_samples,
                            d_rand_state, d_world, d_cam, d_mats,
                            d_prim_indices, d_nodes, d_root);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds.\n";
    // Write to PPM
    CHECK_CUDA(cudaMemcpy(h_screen, d_screen, screen_size, cudaMemcpyDeviceToHost));
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j*image_width + i;
            auto pixel_color = color(h_screen[pixel_index]);
            write_color(std::cout, pixel_color);
        }
    }

    // Free memory
    CHECK_CUDA(cudaFree(d_screen));
    CHECK_CUDA(cudaFree(d_world));
    CHECK_CUDA(cudaFree(d_cam));
    CHECK_CUDA(cudaFree(d_mats));
    CHECK_CUDA(cudaFree(d_nodes));
    CHECK_CUDA(cudaFree(d_prim_indices));
    CHECK_CUDA(cudaFree(d_root));
    CHECK_CUDA(cudaFree(d_node_count));
    CHECK_CUDA(cudaFree(d_rand_state));

    free(h_screen);
    free(h_mats);



    std::clog << "\rDone. \n";
}

