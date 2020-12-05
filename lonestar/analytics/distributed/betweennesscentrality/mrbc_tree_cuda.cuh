#pragma once

#include "galois/runtime/cuda/DeviceSync.h"
#include "gpu_hash_table.cuh"

struct MRBCTreeCUDA {
  using BitSet = DynamicBitset;
  using FlatMap = gpu_hash_table<uint32_t, BitSet, SlabHashTypeT::ConcurrentMap>;

  FlatMap distanceTree;

  uint32_t numNonInfinity;
};
