/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of
 *       conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of
 *       conditions and the following disclaimer in the documentation and/or other materials
 *       provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used
 *       to endorse or promote products derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TOR (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <bvh/common.h>
#include <atomic>
#include <stdexcept>
#include <stdint.h>
#include <string>
#include <vector>

#define CUDA_CHECK_THROW(x)                                                                                               \
    do {                                                                                                                  \
        cudaError_t result = x;                                                                                           \
        if (result != cudaSuccess)                                                                                        \
            throw std::runtime_error(std::string("CUDA Error: " #x " failed with error ") + cudaGetErrorString(result));  \
    } while(0)


namespace bvh {

inline std::atomic<size_t>& total_n_bytes_allocated() {
    static std::atomic<size_t> s_total_n_bytes_allocated{0};
    return s_total_n_bytes_allocated;
}

template<class T>
class GPUMemory {
private:
    T* m_data = nullptr;
    size_t m_size = 0;
    bool m_owned = true;

public:
    GPUMemory() {}

    GPUMemory<T>& operator=(GPUMemory<T>&& other) {
        std::swap(m_data, other.m_data);
        std::swap(m_size, other.m_size);
        return *this;
    }

    GPUMemory(GPUMemory<T>&& other) {
        *this = std::move(other);
    }

    __host__ __device__ GPUMemory(const GPUMemory<T> &other) : m_data{other.m_data}, m_size{other.m_size}, m_owned{false} {}

    void allocate_memory(size_t n_bytes) {
        if (n_bytes == 0) return;
        uint8_t *rawptr = nullptr;
        CUDA_CHECK_THROW(cudaMalloc(&rawptr, n_bytes));
        m_data = (T*)(rawptr);
        total_n_bytes_allocated() += n_bytes;
    }

    void free_memory() {
        if (!m_data) return;
        CUDA_CHECK_THROW(cudaFree(m_data));
        total_n_bytes_allocated() -= get_bytes();
        m_data = nullptr;
    }

    GPUMemory(const size_t size) {
        resize(size);
    }

    __host__ __device__ ~GPUMemory() {
#ifndef __CUDA_ARCH__
        if (!m_owned) return;
        try {
            if (m_data) {
                free_memory();
                m_size = 0;
            }
        } catch (std::runtime_error error) {
            if (std::string{error.what()}.find("driver shutting down") == std::string::npos) {
                fprintf(stderr, "Could not free memory: %s\n", error.what());
            }
        }
#endif
    }

    void resize(const size_t size) {
        if (!m_owned) {
            throw std::runtime_error("Cannot resize non-owned memory.");
        }
        if (m_size != size) {
            if (m_size) {
                free_memory();
            }
            if (size > 0) {
                allocate_memory(size * sizeof(T));
            }
            m_size = size;
        }
    }

    void enlarge(const size_t size) {
        if (size > m_size) {
            resize(size);
        }
    }

    void copy_from_host(const T* host_data, const size_t num_elements) {
        CUDA_CHECK_THROW(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_from_host(const std::vector<T>& data, const size_t num_elements) {
        copy_from_host(data.data(), num_elements);
    }

    void copy_from_host(const T* data) {
        copy_from_host(data, m_size);
    }

    void resize_and_copy_from_host(const T* data, const size_t num_elements) {
        resize(num_elements);
        copy_from_host(data, num_elements);
    }

    void resize_and_copy_from_host(const std::vector<T>& data, const size_t num_elements) {
        resize_and_copy_from_host(data.data(), num_elements);
    }

    void resize_and_copy_from_host(const std::vector<T>& data) {
        resize_and_copy_from_host(data.data(), data.size());
    }

    void copy_to_host(T* host_data, const size_t num_elements) const {
        CUDA_CHECK_THROW(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
    }

    void copy_to_host(std::vector<T>& data) const {
        if (data.size() < m_size) {
            throw std::runtime_error("Vector too small for copy_to_host");
        }
        copy_to_host(data.data(), m_size);
    }

    T* data() const {
        return m_data;
    }

    __host__ __device__ T& operator[](size_t idx) const {
        return m_data[idx];
    }

    size_t get_num_elements() const { return m_size; }
    size_t size() const { return m_size; }
    size_t get_bytes() const { return m_size * sizeof(T); }
    size_t bytes() const { return get_bytes(); }
};

}
