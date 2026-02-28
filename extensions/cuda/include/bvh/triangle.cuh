#pragma once

#include <bvh/common.h>

namespace bvh {

struct Triangle {

    __host__ __device__ Eigen::Vector3f normal() const {
        return (b - a).cross(c - a).normalized();
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd, Eigen::Vector3f& n) const {
        Eigen::Vector3f v1v0 = b - a;
        Eigen::Vector3f v2v0 = c - a;
        Eigen::Vector3f rov0 = ro - a;
        n = v1v0.cross(v2v0);
        Eigen::Vector3f q = rov0.cross(rd);
        float d = 1.0f / rd.dot(n);
        float u = d * -q.dot(v2v0);
        float v = d * q.dot(v1v0);
        float t = d * -n.dot(rov0);
        if (u < 0.0f || u > 1.0f || v < 0.0f || (u + v) > 1.0f || t < 0.0f) t = 1e6f;
        return t;
    }

    __host__ __device__ float ray_intersect(const Eigen::Vector3f &ro, const Eigen::Vector3f &rd) const {
        Eigen::Vector3f n;
        return ray_intersect(ro, rd, n);
    }

    __host__ __device__ Eigen::Vector3f centroid() const {
        return (a + b + c) / 3.0f;
    }

    __host__ __device__ float centroid(int axis) const {
        return (a[axis] + b[axis] + c[axis]) / 3;
    }

    __host__ __device__ void get_vertices(Eigen::Vector3f v[3]) const {
        v[0] = a;
        v[1] = b;
        v[2] = c;
    }

    Eigen::Vector3f a, b, c;
};

}
