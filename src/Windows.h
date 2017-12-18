#pragma once

#ifdef __CUDACC__
#define NvCUDA2(grid, block) <<< grid, block >>>
#define NvCUDA3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define NvCUDA4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define NvCUDA2(grid, block)
#define NvCUDA3(grid, block, sh_mem)
#define NvCUDA4(grid, block, sh_mem, stream)
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cstdint>

//scommentare per utilizzare gli ifdef DEBUG
//#define DEBUG
//#define DEBUG_SUB
#define BIS

#define exit(n) system("pause"); exit(n);