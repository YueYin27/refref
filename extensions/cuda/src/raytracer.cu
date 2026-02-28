#include <bvh/raytracer.h>
#include <bvh/common.h>
#include <bvh/bvh.cuh>

#include <Eigen/Dense>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace bvh {

class RayTracerImpl : public RayTracer {
public:
    RayTracerImpl(Ref<const Verts> vertices, Ref<const Trigs> triangles,
                  const std::vector<int32_t>& tri_geo_ids_vec) : RayTracer()
    {
        const size_t n_triangles = triangles.rows();

        triangles_cpu.resize(n_triangles);
        geo_ids_cpu.resize(n_triangles);

        for (size_t i = 0; i < n_triangles; i++) {
            triangles_cpu[i] = {
                vertices.row(triangles(i, 0)),
                vertices.row(triangles(i, 1)),
                vertices.row(triangles(i, 2))
            };
            geo_ids_cpu[i] = tri_geo_ids_vec[i];
        }

        if (!triangle_bvh) {
            triangle_bvh = TriangleBvh::make();
        }

        // build() reorders both triangles_cpu and geo_ids_cpu in sync
        triangle_bvh->build(triangles_cpu, geo_ids_cpu, 8);

        triangles_gpu.resize_and_copy_from_host(triangles_cpu);
        geo_ids_gpu.resize_and_copy_from_host(geo_ids_cpu);
    }

    void trace(at::Tensor rays_o, at::Tensor rays_d,
               at::Tensor positions, at::Tensor normals,
               at::Tensor depth, at::Tensor geo_ids) override
    {
        const uint32_t n_elements = rays_o.size(0);
        cudaStream_t stream = at::cuda::getCurrentCUDAStream();

        triangle_bvh->ray_trace_gpu(
            n_elements,
            rays_o.data_ptr<float>(),
            rays_d.data_ptr<float>(),
            positions.data_ptr<float>(),
            normals.data_ptr<float>(),
            depth.data_ptr<float>(),
            geo_ids.data_ptr<int32_t>(),
            triangles_gpu.data(),
            geo_ids_gpu.data(),
            stream
        );
    }

    std::vector<Triangle> triangles_cpu;
    std::vector<int32_t> geo_ids_cpu;
    GPUMemory<Triangle> triangles_gpu;
    GPUMemory<int32_t> geo_ids_gpu;
    std::shared_ptr<TriangleBvh> triangle_bvh;
};

RayTracer* create_raytracer(Ref<const Verts> vertices, Ref<const Trigs> triangles,
                             const std::vector<int32_t>& tri_geo_ids) {
    return new RayTracerImpl{vertices, triangles, tri_geo_ids};
}

}
