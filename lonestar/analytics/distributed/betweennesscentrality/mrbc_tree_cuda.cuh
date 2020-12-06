#pragma once

#include "galois/runtime/cuda/DeviceSync.h"
#include "gpu_hash_table.cuh"

using BitSet = DynamicBitset;
using FlatMap = gpu_hash_table<uint32_t, BitSet, SlabHashTypeT::ConcurrentMap>;

struct MRBCTreeCUDA {
  FlatMap distanceTree;

  uint32_t numNonInfinity;
};
