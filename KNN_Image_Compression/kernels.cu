#include <cuda_runtime.h>

#define CHANNELS 3
#define K 16


__global__ void assign_clusters(const unsigned char *pixels, float *centroids, int *labels, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int offset = idx * CHANNELS;
    float min_dist = 1e10f;
    int best_k = 0;

    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;
        for (int c = 0; c < CHANNELS; ++c) {
            float diff = pixels[offset + c] - centroids[k * CHANNELS + c];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_k = k;
        }
    }
    labels[idx] = best_k;
}

__global__ void update_centroids(const unsigned char *pixels, float *centroids, int *labels, int num_pixels, int *counts, float *sums) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int cluster = labels[idx];
    int offset = idx * CHANNELS;

    for (int c = 0; c < CHANNELS; ++c) {
        atomicAdd(&sums[cluster * CHANNELS + c], (float)pixels[offset + c]);
    }
    atomicAdd(&counts[cluster], 1);
}

__global__ void quantize_pixels(unsigned char *pixels, float *centroids, int *labels, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pixels) return;

    int cluster = labels[idx];
    int offset = idx * CHANNELS;

    for (int c = 0; c < CHANNELS; ++c) {
        pixels[offset + c] = (unsigned char)(centroids[cluster * CHANNELS + c]);
    }
}