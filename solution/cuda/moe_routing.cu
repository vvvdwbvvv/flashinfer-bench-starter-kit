// Author:  wei-min chen
// Description:  DeepSeek-V3 "No-aux" routing logic (Sigmoid + Groups + Top-K)
#include <cub/cub.cuh>
#include <cuda_bf16.h>
#include "include/utils.cuh"

struct rankItem {
    float score;
    int idx;

    __device__ __forceinline__ bool operator>(const rankItem& other) const {
        if (score > other.score) return true;
        if (score < other.score) return false;
        return idx < other.idx;
    }
};

struct GreaterRankItem {
    __device__ __forceinline__ bool operator()(const rankItem& a, const rankItem& b) const {
        return (a.score > b.score) || (a.score == b.score && a.idx < b.idx);
    }
};

template<int E_GLOBAL, int E_LOCAL, int TOP_K>
__global__ void router(
    const float* __restrict__ routing_logits, // [T, E_global]
    const __nv_bfloat16* __restrict__ routing_bias,   // [E_global]
    int* expert_token_counts,                 // [E_global]
    int* token_expert_indices,                // [T, TOP_K]
    float* token_expert_weights,              // [T, TOP_K]
    int* token_expert_slots,                  // [T, TOP_K]
    int T, int local_expert_offset, float routed_scaling_factor) {

    int token = blockIdx.x;
    if (token >= T) return;

    constexpr int N_GROUP = 8;
    constexpr int TOPK_GROUP = 4;
    constexpr int GROUP_SIZE = E_GLOBAL / N_GROUP; // 32

    int tid = threadIdx.x;
    int group_id = tid / GROUP_SIZE;
    int lane_id = tid % GROUP_SIZE;

    // --- Shared memory allocations ---
    typedef cub::WarpMergeSort<rankItem, 1, 32> WarpSort32;
    __shared__ typename WarpSort32::TempStorage warp_sort_temp[N_GROUP];

    typedef cub::WarpMergeSort<rankItem, 2, 32> WarpSort64;
    __shared__ typename WarpSort64::TempStorage final_sort_temp;

    __shared__ float s_group_scores[N_GROUP];
    __shared__ bool group_mask[N_GROUP];
    __shared__ rankItem shared_candidates[64];
    __shared__ int final_indices[8];


    // Initialization
    if (tid < 8) final_indices[tid] = -1;
    __syncthreads();

    // 1. Warp Sort for Group Top-2
    float logit = routing_logits[token * E_GLOBAL + tid];
    float s_wb = sigmoid(logit + __bfloat162float(routing_bias[tid]));

    rankItem item_arr[1];
    item_arr[0].score = s_wb;
    item_arr[0].idx = tid;

    WarpSort32(warp_sort_temp[group_id]).Sort(item_arr, GreaterRankItem());

    float group_top1 = __shfl_sync(0xffffffff, item_arr[0].score, 0);
    float group_top2 = __shfl_sync(0xffffffff, item_arr[0].score, 1);

    if (lane_id == 0) {
        s_group_scores[group_id] = group_top1 + group_top2;
    }
    __syncthreads();

    // 2. Select Top-4 groups
    if (tid < N_GROUP) {
        float my_score = s_group_scores[tid];
        int rank = 0;
        #pragma unroll
        for (int i = 0; i < N_GROUP; ++i) {
            if (s_group_scores[i] > my_score || (s_group_scores[i] == my_score && i < tid)) rank++;
        }
        group_mask[tid] = (rank < TOPK_GROUP);
    }
    __syncthreads();

    // 3. Select Global Top-K from 64 candidates
    item_arr[0].score = (group_mask[group_id]) ? s_wb : -1e20f;
    item_arr[0].idx = tid;

    WarpSort32(warp_sort_temp[group_id]).Sort(item_arr, GreaterRankItem());

    if (lane_id < 8) {
        shared_candidates[group_id * 8 + lane_id] = item_arr[0];
    }
    __syncthreads();

    // Warp 0 does the final reduction
    if (tid < 32) {
        rankItem candidates[2];
        candidates[0] = shared_candidates[tid];
        candidates[1] = shared_candidates[tid + 32];

        WarpSort64(final_sort_temp).Sort(candidates, GreaterRankItem());

        // threads 0-3 have top-8
        float winner_s_arr[2] = {0.0f, 0.0f};
        if (tid < 4) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                if (candidates[i].score > -1e10f) {
                    winner_s_arr[i] = sigmoid(routing_logits[token * E_GLOBAL + candidates[i].idx]);
                }
            }
        }

        float partial_sum = winner_s_arr[0] + winner_s_arr[1];
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            partial_sum += __shfl_xor_sync(0xffffffff, partial_sum, offset);
        }
        
        float total_sigmoid_sum = partial_sum + 1e-20f;

        if (tid < 4) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                if (candidates[i].score > -1e10f) {
                    int pos = tid * 2 + i;
                    float weight = (winner_s_arr[i] / total_sigmoid_sum) * routed_scaling_factor;
                    token_expert_indices[token * TOP_K + pos] = candidates[i].idx;
                    token_expert_weights[token * TOP_K + pos] = weight;
                    final_indices[pos] = candidates[i].idx;
                }
            }
        }
    }
    __syncthreads();

    // 4. Expert Counting & Slot Assignment via atomicAdd
    if (tid < TOP_K && final_indices[tid] != -1) {
        int expert_id = final_indices[tid];
        int slot = atomicAdd(&expert_token_counts[expert_id], 1);
        token_expert_slots[token * TOP_K + tid] = slot;
    }
}