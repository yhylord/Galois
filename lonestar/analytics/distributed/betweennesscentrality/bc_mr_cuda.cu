#include "gg.h"
#include "ggcuda.h"
#include "cub/cub.cuh"
#include "cub/util_allocator.cuh"
#include "bc_mr_cuda.cuh"

#include <vector>

void InitializeGraph_allNodes_cuda(CUDA_Context* cuda_ctx) {
}

void InitializeIteration_allNodes_cuda(uint32_t infinity, const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx) {}

void FindMessageToSync_allNodes_cuda(uint32_t &dga, uint32_t roundNumber, CUDA_Context* cuda_ctx) {}

void ConfirmMessageToSend_allNodes_cuda(uint32_t roundNumber, CUDA_Context* cuda_ctx) {}

void SendAPSPMessages_nodesWithEdges_cuda(uint32_t &dga, CUDA_Context* cuda_ctx) {}

void RoundUpdate_allNodes_cuda(CUDA_Context* cuda_ctx) {} 

void BackFindMessageToSend_allNodes_cuda(uint32_t roundNumber, uint32_t lastRoundNumber, CUDA_Context* cuda_ctx) {}

void BackProp_nodesWithEdges_cuda(uint32_t lastRoundNumber, CUDA_Context* cuda_ctx) {}

void BC_masterNodes_cuda(const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx) {}

void Sanity_masterNodes_cuda(float &dga_max, float &dga_min, float &dga_sum, CUDA_Context* cuda_ctx) {}
