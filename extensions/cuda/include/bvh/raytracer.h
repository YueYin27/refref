#pragma once

#include <Eigen/Dense>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <cuda_runtime.h>

using namespace Eigen;

using Verts = Matrix<float, Dynamic, 3, RowMajor>;
using Trigs = Matrix<uint32_t, Dynamic, 3, RowMajor>;

namespace bvh {

class RayTracer {
public:
    RayTracer() {}
    virtual ~RayTracer() {}

    virtual void trace(at::Tensor rays_o, at::Tensor rays_d,
                       at::Tensor positions, at::Tensor normals,
                       at::Tensor depth, at::Tensor geo_ids) = 0;
};

RayTracer* create_raytracer(Ref<const Verts> vertices, Ref<const Trigs> triangles,
                             const std::vector<int32_t>& tri_geo_ids);

}
