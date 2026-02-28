#pragma once

#include <bvh/common.h>
#include <bvh/triangle.cuh>

namespace bvh {

struct BoundingBox {

    Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());

    __host__ __device__ BoundingBox() {}

    __host__ __device__ BoundingBox(const Eigen::Vector3f& a, const Eigen::Vector3f& b) : min{a}, max{b} {}

    __host__ __device__ explicit BoundingBox(const Triangle& tri) {
        min = max = tri.a;
        enlarge(tri.b);
        enlarge(tri.c);
    }

    BoundingBox(std::vector<Triangle>::iterator begin, std::vector<Triangle>::iterator end) {
        min = max = begin->a;
        for (auto it = begin; it != end; ++it) {
            enlarge(*it);
        }
    }

    __host__ __device__ void enlarge(const BoundingBox& other) {
        min = min.cwiseMin(other.min);
        max = max.cwiseMax(other.max);
    }

    __host__ __device__ void enlarge(const Triangle& tri) {
        enlarge(tri.a);
        enlarge(tri.b);
        enlarge(tri.c);
    }

    __host__ __device__ void enlarge(const Eigen::Vector3f& point) {
        min = min.cwiseMin(point);
        max = max.cwiseMax(point);
    }

    __host__ __device__ Eigen::Vector3f diag() const {
        return max - min;
    }

    __host__ __device__ Eigen::Vector3f center() const {
        return 0.5f * (max + min);
    }

    __host__ __device__ bool is_empty() const {
        return (max.array() < min.array()).any();
    }

    __host__ __device__ Eigen::Vector2f ray_intersect(Eigen::Ref<const Eigen::Vector3f> pos, Eigen::Ref<const Eigen::Vector3f> dir) const {
        float tmin = (min.x() - pos.x()) / dir.x();
        float tmax = (max.x() - pos.x()) / dir.x();

        if (tmin > tmax) {
            host_device_swap(tmin, tmax);
        }

        float tymin = (min.y() - pos.y()) / dir.y();
        float tymax = (max.y() - pos.y()) / dir.y();

        if (tymin > tymax) {
            host_device_swap(tymin, tymax);
        }

        if (tmin > tymax || tymin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tymin > tmin) tmin = tymin;
        if (tymax < tmax) tmax = tymax;

        float tzmin = (min.z() - pos.z()) / dir.z();
        float tzmax = (max.z() - pos.z()) / dir.z();

        if (tzmin > tzmax) {
            host_device_swap(tzmin, tzmax);
        }

        if (tmin > tzmax || tzmin > tmax) {
            return { std::numeric_limits<float>::max(), std::numeric_limits<float>::max() };
        }

        if (tzmin > tmin) tmin = tzmin;
        if (tzmax < tmax) tmax = tzmax;

        return { tmin, tmax };
    }
};

}
