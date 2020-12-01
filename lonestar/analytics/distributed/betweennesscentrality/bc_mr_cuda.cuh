#pragma once
#include <cuda.h>
#include <stdio.h>
#include <unistd.h>
#include "bc_mr_cuda.h"
#include "galois/runtime/cuda/DeviceSync.h"

struct CUDA_Context : public CUDA_Context_Common {
  // BCData
  struct CUDA_Context_Field<uint32_t> minDistance;
  struct CUDA_Context_Field<ShortPathType> shortPathCount;
  struct CUDA_Context_Field<float> dependency;
  // fixme: need to have a MRBCTree equivalent field
  // MRBCTree dTree;
  struct CUDA_Context_Field<float> bc;
  struct CUDA_Context_Field<uint32_t> roundIndexToSend;
};

struct CUDA_Context* get_CUDA_context(int id) {
  struct CUDA_Context* ctx;
  ctx = (struct CUDA_Context* ) calloc(1, sizeof(struct CUDA_Context));
  ctx->id = id;
  return ctx;
}

bool init_CUDA_context(struct CUDA_Context* ctx, int device) {
  return init_CUDA_context_common(ctx, device);
}

void load_graph_CUDA(struct CUDA_Context* ctx, MarshalGraph &g, unsigned num_hosts) {
  // fixme: load fields related to MRBCTree
  size_t mem_usage = mem_usage_CUDA_common(g, num_hosts);
  mem_usage += mem_usage_CUDA_field(&ctx->minDistance, g, num_hosts);
  mem_usage += mem_usage_CUDA_field(&ctx->shortPathCount, g, num_hosts);
  mem_usage += mem_usage_CUDA_field(&ctx->dependency, g, num_hosts);
  mem_usage += mem_usage_CUDA_field(&ctx->bc, g, num_hosts);
  mem_usage += mem_usage_CUDA_field(&ctx->roundIndexToSend, g, num_hosts);
  printf("[%d] Host memory for communication context: %3u MB\n", ctx->id, mem_usage/1048756);
  load_graph_CUDA_common(ctx, g, num_hosts);
  load_graph_CUDA_field(ctx, &ctx->minDistance, num_hosts);
  load_graph_CUDA_field(ctx, &ctx->shortPathCount, num_hosts);
  load_graph_CUDA_field(ctx, &ctx->dependency, num_hosts);
  load_graph_CUDA_field(ctx, &ctx->bc, num_hosts);
  load_graph_CUDA_field(ctx, &ctx->roundIndexToSend, num_hosts);
  reset_CUDA_context(ctx);
}

void reset_CUDA_context(struct CUDA_Context* ctx) {
  ctx->minDistance.data.zero_gpu();
  ctx->shortPathCount.data.zero_gpu();
  ctx->dependency.data.zero_gpu();
  // fixme: need to reset MRBCTree fields
  ctx->bc.data.zero_gpu();
  ctx->roundIndexToSend.data.zero_gpu();
}

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx) {
  ctx->dependency.is_updated.cpu_rd_ptr()->reset();
}
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
  reset_bitset_field(&ctx->dependency, begin, end);
}
void get_bitset_dependency_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
  ctx->dependency.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}

void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx) {
  ctx->minDistance.is_updated.cpu_rd_ptr()->reset();
}
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end) {
  reset_bitset_field(&ctx->minDistance, begin, end);
}
void get_bitset_minDistances_cuda(struct CUDA_Context* ctx, uint64_t* bitset_compute) {
  ctx->minDistance.is_updated.cpu_rd_ptr()->copy_to_cpu(bitset_compute);
}
