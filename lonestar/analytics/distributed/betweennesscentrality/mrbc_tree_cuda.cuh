#pragma once

#include "galois/runtime/cuda/DeviceSync.h"

using BitSet = DynamicBitset;
using FlatMap =

struct MRBCTreeCUDA {
  FlatMap distanceTree;

  uint32_t numNonInfinity;
public:
  __device__ void setDistance(uint32_t index, uint32_t newDistance) {
  }
};
