#pragma once

#include <bvh/common.h>
#include <bvh/triangle.cuh>
#include <bvh/bounding_box.cuh>
#include <bvh/gpu_memory.h>

#include <memory>

namespace bvh {

struct TriangleBvhNode {
    BoundingBox bb;
    int left_idx;  // negative values indicate leaves
    int right_idx;
};

template <typename T, int MAX_SIZE=32>
class FixedStack {
public:
    __host__ __device__ void push(T val) {
        if (m_count >= MAX_SIZE-1) {
            printf("WARNING TOO BIG\n");
        }
        m_elems[m_count++] = val;
    }

    __host__ __device__ T pop() {
        return m_elems[--m_count];
    }

    __host__ __device__ bool empty() const {
        return m_count <= 0;
    }

private:
    T m_elems[MAX_SIZE];
    int m_count = 0;
};

using FixedIntStack = FixedStack<int>;

class TriangleBvh {

protected:
    std::vector<TriangleBvhNode> m_nodes;
    GPUMemory<TriangleBvhNode> m_nodes_gpu;
    TriangleBvh() {};

public:
    virtual void build(std::vector<Triangle>& triangles,
                       std::vector<int32_t>& geo_ids,
                       uint32_t n_primitives_per_leaf) = 0;

    virtual void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d,
                               float* positions, float* normals, float* depth,
                               int32_t* geo_ids,
                               const Triangle* gpu_triangles, const int32_t* gpu_tri_geo_ids,
                               cudaStream_t stream) = 0;

    static std::unique_ptr<TriangleBvh> make();

    TriangleBvhNode* nodes_gpu() const {
        return m_nodes_gpu.data();
    }
};

}
