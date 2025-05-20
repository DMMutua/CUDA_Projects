#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHANNELS 3
#define K 16
#define MAX_ITER 10
#define BLOCK_SIZE 256

// Func. declarations
void initialize_centroids(float *centroids, const unsigned char *pixels, int num_pixels);
unsigned char* load_png(const char *filename, int *width, int *height);
__global__ void assign_clusters(const unsigned char *pixels, float *centroids, int *labels, int num_pixels);
__global__ void update_centroids(const unsigned char *pixels, float *centroids, int *labels, int num_pixels, int *counts, float *sums);
__global__ void quantize_pixels(unsigned char *pixels, float *centroids, int *labels, int num_pixels);

// Entry Point ...
int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_png>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int width, height;
    unsigned char *image = load_png(argv[1], &width, &height);

    printf("Loaded %s (%dx%d)\n", width, height);
    compress_image(image, width, height);

    // TODO: ```save_png```

    free(image);
    return 0;
}