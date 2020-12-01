#pragma once

using ShortPathType = double;

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
void get_bitset_dependency_cuda(struct CUDA_Context *ctx, uint64_t* bitset_compute);

void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
void get_bitset_minDistances_cuda(struct CUDA_Context *ctx, uint64_t* bitset_compute);
