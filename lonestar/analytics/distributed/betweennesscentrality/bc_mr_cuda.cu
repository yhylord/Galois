#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "bc_mr_cuda.cuh"

#include <vector>

const uint32_t TB_SIZE = 256;

void kernel_sizing(CSRGraph &, dim3 &, dim3 &);

__global__ void InitializeGraph(CSRGraph graph, uint32_t __begin, uint32_t __end,
    uint32_t *p_minDistance,
    ShortPathType *p_shortPathCount,
    float *p_dependency,
    float *p_bc,
    uint32_t *p_roundIndexToSend) {
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  index_type src_end;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      p_bc[src] = 0;
    }
  }
}

template <typename Data>
void amplify(Data& data, uint32_t multiple) {
  auto oldSize = data.size();
  data.free();
  data.alloc(oldSize * multiple);
}

template <typename Type>
void amplify(CUDA_Context_Field<Type>* field, uint32_t multiple) {
  amplify(field->data, multiple);
  amplify(field->is_updated, multiple);
  amplify(field->shared_data, multiple);
}

void InitializeGraph_cuda(uint32_t __begin, uint32_t __end, uint32_t vectorSize, CUDA_Context* ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);
  amplify(&ctx->minDistance, vectorSize);
  amplify(&ctx->shortPathCount, vectorSize);
  amplify(&ctx->dependency, vectorSize);
  InitializeGraph<<<blocks, threads>>>(ctx->gg, __begin, __end,
      ctx->minDistance.data.gpu_wr_ptr(),
      ctx->shortPathCount.data.gpu_wr_ptr(),
      ctx->dependency.data.gpu_wr_ptr(),
      ctx->bc.data.gpu_wr_ptr(),
      ctx->roundIndexToSend.data.gpu_wr_ptr()
      );
  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void InitializeGraph_allNodes_cuda(uint32_t vectorSize, CUDA_Context* ctx) {
  InitializeGraph_cuda(0, ctx->gg.nnodes, vectorSize, ctx);
}

// TODO: Pass in MRBC Tree
__global__ void InitializeIteration(uint32_t __begin, uint32_t __end,
    uint32_t infinity, uint32_t numSourcesPerRound, uint64_t* nodesToConsider, 
    uint32_t *p_minDistance,
    ShortPathType *p_shortPathCount,
    float *p_dependency,
    uint32_t *p_roundIndexToSend
    ) {
}

void InitializeIteration_allNodes_cuda(uint32_t infinity, uint32_t numSourcesPerRound, const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  auto size = nodesToConsider.size() * sizeof(uint64_t);
  uint64_t* nodes;
  cudaMalloc((void**)&nodes, size);
  cudaMemcpy(nodes, nodesToConsider.data(), size, cudaMemcpyDeviceToHost);

  InitializeIteration<<<blocks, threads>>>(0, cuda_ctx->gg.nnodes, infinity, numSourcesPerRound, nodes,
      cuda_ctx->minDistance.data.gpu_wr_ptr(),
      cuda_ctx->shortPathCount.data.gpu_wr_ptr(),
      cuda_ctx->dependency.data.gpu_wr_ptr(),
      cuda_ctx->roundIndexToSend.data.gpu_wr_ptr()
      );
}

void FindMessageToSync_allNodes_cuda(uint32_t &dga, uint32_t roundNumber, CUDA_Context* cuda_ctx) {}

void ConfirmMessageToSend_allNodes_cuda(uint32_t roundNumber, CUDA_Context* cuda_ctx) {}

void SendAPSPMessages_nodesWithEdges_cuda(uint32_t &dga, CUDA_Context* cuda_ctx) {}

void RoundUpdate_allNodes_cuda(CUDA_Context* cuda_ctx) {} 

void BackFindMessageToSend_allNodes_cuda(uint32_t roundNumber, uint32_t lastRoundNumber, CUDA_Context* cuda_ctx) {}

void BackProp_nodesWithEdges_cuda(uint32_t lastRoundNumber, CUDA_Context* cuda_ctx) {}

void BC_masterNodes_cuda(const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx) {
  auto size = nodesToConsider.size() * sizeof(uint64_t);
  uint64_t* nodes;
  cudaMalloc((void**)&nodes, size);
  cudaMemcpy(nodes, nodesToConsider.data(), size, cudaMemcpyDeviceToHost);
}

void Sanity_masterNodes_cuda(float &dga_max, float &dga_min, float &dga_sum, CUDA_Context* cuda_ctx) {}
