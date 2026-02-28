#include <bvh/common.h>
#include <bvh/triangle.cuh>
#include <bvh/bvh.cuh>

#include <stack>
#include <iostream>

using namespace Eigen;
using namespace bvh;

namespace bvh {

constexpr float MAX_DIST = 10.0f;

struct DistAndIdx {
    float dist;
    uint32_t idx;

    __host__ __device__ bool operator<(const DistAndIdx& other) {
        return dist < other.dist;
    }
};

template <typename T>
__host__ __device__ void inline compare_and_swap(T& t1, T& t2) {
    if (t1 < t2) {
        T tmp{t1}; t1 = t2; t2 = tmp;
    }
}

template <uint32_t N, typename T>
__host__ __device__ void sorting_network(T values[N]) {
    static_assert(N <= 8, "Sorting networks are only implemented up to N==8");
    if (N <= 1) {
        return;
    } else if (N == 2) {
        compare_and_swap(values[0], values[1]);
    } else if (N == 3) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 4) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[1], values[2]);
    } else if (N == 5) {
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[2], values[3]);
    } else if (N == 6) {
        compare_and_swap(values[0], values[5]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[0], values[3]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
    } else if (N == 7) {
        compare_and_swap(values[0], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[5]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    } else if (N == 8) {
        compare_and_swap(values[0], values[2]);
        compare_and_swap(values[1], values[3]);
        compare_and_swap(values[4], values[6]);
        compare_and_swap(values[5], values[7]);
        compare_and_swap(values[0], values[4]);
        compare_and_swap(values[1], values[5]);
        compare_and_swap(values[2], values[6]);
        compare_and_swap(values[3], values[7]);
        compare_and_swap(values[0], values[1]);
        compare_and_swap(values[2], values[3]);
        compare_and_swap(values[4], values[5]);
        compare_and_swap(values[6], values[7]);
        compare_and_swap(values[2], values[4]);
        compare_and_swap(values[3], values[5]);
        compare_and_swap(values[1], values[4]);
        compare_and_swap(values[3], values[6]);
        compare_and_swap(values[1], values[2]);
        compare_and_swap(values[3], values[4]);
        compare_and_swap(values[5], values[6]);
    }
}

__global__ void raytrace_kernel(
    uint32_t n_elements,
    const Vector3f* __restrict__ rays_o,
    const Vector3f* __restrict__ rays_d,
    Vector3f* __restrict__ positions,
    Vector3f* __restrict__ normals,
    float* __restrict__ depth,
    int32_t* __restrict__ geo_ids,
    const TriangleBvhNode* __restrict__ nodes,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ tri_geo_ids);

struct TaggedTriangle {
    Triangle tri;
    int32_t geo_id;
};

template <uint32_t BRANCHING_FACTOR>
class TriangleBvhWithBranchingFactor : public TriangleBvh {
public:
    __host__ __device__ static std::pair<int, float> ray_intersect(
        Ref<const Vector3f> ro, Ref<const Vector3f> rd,
        const TriangleBvhNode* __restrict__ bvhnodes,
        const Triangle* __restrict__ triangles)
    {
        FixedIntStack query_stack;
        query_stack.push(0);

        float mint = MAX_DIST;
        int shortest_idx = -1;

        while (!query_stack.empty()) {
            int idx = query_stack.pop();
            const TriangleBvhNode& node = bvhnodes[idx];

            if (node.left_idx < 0) {
                int end = -node.right_idx - 1;
                for (int i = -node.left_idx - 1; i < end; ++i) {
                    float t = triangles[i].ray_intersect(ro, rd);
                    if (t < mint) {
                        mint = t;
                        shortest_idx = i;
                    }
                }
            } else {
                DistAndIdx children[BRANCHING_FACTOR];
                uint32_t first_child = node.left_idx;

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    children[i] = {bvhnodes[i + first_child].bb.ray_intersect(ro, rd).x(), i + first_child};
                }

                sorting_network<BRANCHING_FACTOR>(children);

                #pragma unroll
                for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                    if (children[i].dist < mint) {
                        query_stack.push(children[i].idx);
                    }
                }
            }
        }

        return {shortest_idx, mint};
    }

    void ray_trace_gpu(uint32_t n_elements, const float* rays_o, const float* rays_d,
                       float* positions, float* normals, float* depth,
                       int32_t* geo_ids,
                       const Triangle* gpu_triangles, const int32_t* gpu_tri_geo_ids,
                       cudaStream_t stream) override
    {
        const Vector3f* rays_o_vec = (const Vector3f*)rays_o;
        const Vector3f* rays_d_vec = (const Vector3f*)rays_d;
        Vector3f* positions_vec = (Vector3f*)positions;
        Vector3f* normals_vec = (Vector3f*)normals;

        linear_kernel(raytrace_kernel, 0, stream,
            n_elements,
            rays_o_vec,
            rays_d_vec,
            positions_vec,
            normals_vec,
            depth,
            geo_ids,
            m_nodes_gpu.data(),
            gpu_triangles,
            gpu_tri_geo_ids
        );
    }

    void build(std::vector<Triangle>& triangles,
               std::vector<int32_t>& geo_ids,
               uint32_t n_primitives_per_leaf) override
    {
        m_nodes.clear();

        const size_t n_tris = triangles.size();
        std::vector<TaggedTriangle> tagged(n_tris);
        for (size_t i = 0; i < n_tris; i++) {
            tagged[i].tri = triangles[i];
            tagged[i].geo_id = geo_ids[i];
        }

        m_nodes.emplace_back();
        // Compute root bounding box
        {
            BoundingBox bb;
            bb.min = bb.max = tagged[0].tri.a;
            for (size_t i = 0; i < n_tris; i++) {
                bb.enlarge(tagged[i].tri);
            }
            m_nodes.front().bb = bb;
        }

        struct BuildNode {
            int node_idx;
            size_t begin;
            size_t end;
        };

        std::stack<BuildNode> build_stack;
        build_stack.push({0, 0, n_tris});

        while (!build_stack.empty()) {
            const BuildNode curr = build_stack.top();
            build_stack.pop();
            size_t node_idx = curr.node_idx;

            struct ChildRange {
                size_t begin, end;
                int node_idx;
            };

            std::array<ChildRange, BRANCHING_FACTOR> children;
            children[0].begin = curr.begin;
            children[0].end = curr.end;

            int n_children = 1;
            while (n_children < (int)BRANCHING_FACTOR) {
                for (int i = n_children - 1; i >= 0; --i) {
                    auto& child = children[i];
                    size_t count = child.end - child.begin;

                    if (count <= 1) {
                        size_t orig_begin = child.begin;
                        size_t orig_end = child.end;
                        children[i * 2].begin = orig_begin;
                        children[i * 2].end = orig_end;
                        children[i * 2 + 1].begin = orig_begin;
                        children[i * 2 + 1].end = orig_end;
                        continue;
                    }

                    Vector3f mean = Vector3f::Zero();
                    for (size_t j = child.begin; j < child.end; j++) {
                        mean += tagged[j].tri.centroid();
                    }
                    mean /= (float)count;

                    Vector3f var = Vector3f::Zero();
                    for (size_t j = child.begin; j < child.end; j++) {
                        Vector3f diff = tagged[j].tri.centroid() - mean;
                        var += diff.cwiseProduct(diff);
                    }
                    var /= (float)count;

                    Vector3f::Index axis;
                    var.maxCoeff(&axis);

                    auto it_begin = tagged.begin() + child.begin;
                    auto it_end = tagged.begin() + child.end;
                    size_t half = std::max(count / 2, (size_t)1);
                    auto m = it_begin + half;
                    std::nth_element(it_begin, m, it_end,
                        [&](const TaggedTriangle& a, const TaggedTriangle& b) {
                            return a.tri.centroid(axis) < b.tri.centroid(axis);
                        });

                    size_t mid = child.begin + half;
                    size_t orig_begin = child.begin;
                    size_t orig_end = child.end;
                    children[i * 2].begin = orig_begin;
                    children[i * 2].end = mid;
                    children[i * 2 + 1].begin = mid;
                    children[i * 2 + 1].end = orig_end;
                }
                n_children *= 2;
            }

            m_nodes[node_idx].left_idx = (int)m_nodes.size();
            for (uint32_t i = 0; i < BRANCHING_FACTOR; ++i) {
                auto& child = children[i];
                assert(child.begin < child.end);
                child.node_idx = (int)m_nodes.size();

                m_nodes.emplace_back();
                BoundingBox bb;
                bb.min = bb.max = tagged[child.begin].tri.a;
                for (size_t j = child.begin; j < child.end; j++) {
                    bb.enlarge(tagged[j].tri);
                }
                m_nodes.back().bb = bb;

                if ((child.end - child.begin) <= n_primitives_per_leaf) {
                    m_nodes.back().left_idx = -(int)child.begin - 1;
                    m_nodes.back().right_idx = -(int)child.end - 1;
                } else {
                    build_stack.push({child.node_idx, child.begin, child.end});
                }
            }
            m_nodes[node_idx].right_idx = (int)m_nodes.size();
        }

        // Write back reordered triangles and geo_ids
        triangles.resize(n_tris);
        geo_ids.resize(n_tris);
        for (size_t i = 0; i < n_tris; i++) {
            triangles[i] = tagged[i].tri;
            geo_ids[i] = tagged[i].geo_id;
        }

        m_nodes_gpu.resize_and_copy_from_host(m_nodes);
    }

    TriangleBvhWithBranchingFactor() {}
};

using TriangleBvh4 = TriangleBvhWithBranchingFactor<4>;

std::unique_ptr<TriangleBvh> TriangleBvh::make() {
    return std::unique_ptr<TriangleBvh>(new TriangleBvh4());
}

__global__ void raytrace_kernel(
    uint32_t n_elements,
    const Vector3f* __restrict__ rays_o,
    const Vector3f* __restrict__ rays_d,
    Vector3f* __restrict__ positions,
    Vector3f* __restrict__ normals,
    float* __restrict__ depth,
    int32_t* __restrict__ geo_ids,
    const TriangleBvhNode* __restrict__ nodes,
    const Triangle* __restrict__ triangles,
    const int32_t* __restrict__ tri_geo_ids)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elements) return;

    Vector3f ro = rays_o[i];
    Vector3f rd = rays_d[i];

    auto p = TriangleBvh4::ray_intersect(ro, rd, nodes, triangles);

    depth[i] = p.second;
    positions[i] = ro + p.second * rd;

    if (p.first >= 0) {
        normals[i] = triangles[p.first].normal();
        geo_ids[i] = tri_geo_ids[p.first];
    } else {
        normals[i].setZero();
        geo_ids[i] = -1;
    }
}

}
