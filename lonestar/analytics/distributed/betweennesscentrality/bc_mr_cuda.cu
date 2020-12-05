#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "mrbc_tree_cuda.cuh"
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
__global__ void InitializeIteration(CSRGraph graph, uint32_t __begin, uint32_t __end,
    uint32_t infinity, uint32_t numSourcesPerRound, uint32_t vectorSize,
    uint64_t* nodesToConsider, 
    uint32_t *p_minDistance,
    ShortPathType *p_shortPathCount,
    float *p_dependency,
    uint32_t *p_roundIndexToSend
    ) {
  unsigned tid = TID_1D;
  unsigned nthreads = TOTAL_THREADS_1D;

  const unsigned __kernel_tb_size = TB_SIZE;
  bool is_source;
  index_type src_end;
  // FP: "1 -> 2;
  // FP: "2 -> 3;
  src_end = __end;
  for (index_type src = __begin + tid; src < src_end; src += nthreads)
  {
    bool pop  = src < __end;
    if (pop)
    {
      for (unsigned i = 0; i < numSourcesPerRound; ++i) {
        is_source = graph.node_data[src] == nodesToConsider[i];
        auto gridIdx = src * vectorSize + i; 
        if (!is_source)
        {
          p_minDistance[gridIdx]     = infinity;
          p_shortPathCount[gridIdx] = 0;
          // fixme: need to set MRBCTree distance for i to 0
          // p_tree[gridIdx] = 0;
        }
        else
        {
          p_minDistance[gridIdx]     = 0;
          p_shortPathCount[gridIdx] = 1;
        }
        p_dependency[gridIdx]       = 0;
      }
    }
  }
}

void InitializeIteration_allNodes_cuda(uint32_t infinity, uint32_t numSourcesPerRound, uint32_t vectorSize, const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  auto size = nodesToConsider.size() * sizeof(uint64_t);
  uint64_t* nodes;
  cudaMalloc((void**)&nodes, size);
  cudaMemcpy(nodes, nodesToConsider.data(), size, cudaMemcpyHostToDevice);

  InitializeIteration<<<blocks, threads>>>(cuda_ctx->gg, 0, cuda_ctx->gg.nnodes, infinity, numSourcesPerRound, vectorSize, nodes,
      cuda_ctx->minDistance.data.gpu_wr_ptr(),
      cuda_ctx->shortPathCount.data.gpu_wr_ptr(),
      cuda_ctx->dependency.data.gpu_wr_ptr(),
      cuda_ctx->roundIndexToSend.data.gpu_wr_ptr()
      );

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

__global__ void FindMessageToSync(CSRGraph graph, uint32_t __begin, uint32_t __end, uint32_t &dga, uint32_t roundNumber,
    uint32_t *p_roundIndexToSend,
    uint32_t *p_minDistance) {}

void FindMessageToSync_allNodes_cuda(uint32_t &dga, uint32_t roundNumber, CUDA_Context* ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  FindMessageToSync<<<blocks, threads>>>(ctx->gg, 0, ctx->gg.nnodes, dga, roundNumber,
      ctx->roundIndexToSend.data.gpu_wr_ptr(),
      ctx->minDistance.data.gpu_wr_ptr());

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void ConfirmMessageToSend_allNodes_cuda(uint32_t roundNumber, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void SendAPSPMessages_nodesWithEdges_cuda(uint32_t &dga, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void RoundUpdate_allNodes_cuda(CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  cudaDeviceSynchronize();
  check_cuda_kernel;
}


void BackFindMessageToSend_allNodes_cuda(uint32_t roundNumber, uint32_t lastRoundNumber, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void BackProp_nodesWithEdges_cuda(uint32_t lastRoundNumber, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void BC_masterNodes_cuda(const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  auto size = nodesToConsider.size() * sizeof(uint64_t);
  uint64_t* nodes;
  cudaMalloc((void**)&nodes, size);
  cudaMemcpy(nodes, nodesToConsider.data(), size, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  check_cuda_kernel;
}

void Sanity_masterNodes_cuda(float &dga_max, float &dga_min, float &dga_sum, CUDA_Context* cuda_ctx) {
  dim3 blocks;
  dim3 threads;
  kernel_sizing(blocks, threads);

  cudaDeviceSynchronize();
  check_cuda_kernel;
}
