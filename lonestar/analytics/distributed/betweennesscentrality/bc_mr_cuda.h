#pragma once

using ShortPathType = double;

void bitset_dependency_reset_cuda(struct CUDA_Context* ctx);
void bitset_dependency_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
void get_bitset_dependency_cuda(struct CUDA_Context *ctx, uint64_t* bitset_compute);

void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx);
void bitset_minDistances_reset_cuda(struct CUDA_Context* ctx, size_t begin, size_t end);
void get_bitset_minDistances_cuda(struct CUDA_Context *ctx, uint64_t* bitset_compute);

float get_node_betweenness_centrality_cuda(CUDA_Context* ctx, uint32_t LID);

void InitializeGraph_allNodes_cuda(uint32_t vectorSize, CUDA_Context* cuda_ctx);

void InitializeIteration_allNodes_cuda(uint32_t infinity, const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx);

void FindMessageToSync_allNodes_cuda(uint32_t &dga, uint32_t roundNumber, CUDA_Context* cuda_ctx);

void ConfirmMessageToSend_allNodes_cuda(uint32_t roundNumber, CUDA_Context* cuda_ctx);

void SendAPSPMessages_nodesWithEdges_cuda(uint32_t &dga, CUDA_Context* cuda_ctx);

void RoundUpdate_allNodes_cuda(CUDA_Context* cuda_ctx);

void BackFindMessageToSend_allNodes_cuda(uint32_t roundNumber, uint32_t lastRoundNumber, CUDA_Context* cuda_ctx);

void BackProp_nodesWithEdges_cuda(uint32_t lastRoundNumber, CUDA_Context* cuda_ctx);

void BC_masterNodes_cuda(const std::vector<uint64_t>& nodesToConsider, CUDA_Context* cuda_ctx);

void Sanity_masterNodes_cuda(float &dga_max, float &dga_min, float &dga_sum, CUDA_Context* cuda_ctx);
