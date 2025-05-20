#include <stdio.h>
#include <stdlib.h>
#include <png.h>
#include <cuda_runtime.h>

#define CHANNELS 3
#define K 16

#define cudaCheckError() { \
    cudaError_t e = cudaGetLastError(); \
    if (e != cudaSuccess) { \
      printf("CUDA Error: %s\n", cudaGetErrorString(e)); \
      exit(EXIT_FAILURE); \
    } \
  }


void initialize_centroids(float *centroids, const unsigned char *pixels, int num_pixels) {
    srand(33);
    for (int k = 0; k < K; ++k) {
        int idx = rand() % num_pixels;
        for (int c = 0; c < CHANNELS; ++c) {
            centroids[k * CHANNELS + c] = (float)pixels[idx * CHANNELS + c];
        }
    }
}

unsigned char* load_png(const char *filename, int *width, int *height) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        perror("File Opening Failed");
        exit(EXIT_FAILURE);
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) exit(EXIT_FAILURE);

    png_infop info = png_create_info_struct(png);
    if (!info) exit(EXIT_FAILURE);

    if (setjmp(png_jmpbuf(png))) exit(EXIT_FAILURE);

    png_init_io(png, fp);
    png_read_info(png, info);

    *width = png_get_image_width(png, info);
    *height = png_get_image_height(png, info);

    png_byte color_type = png_get_color_type(png, info);
    png_byte bit_depth  = png_get_bit_depth(png, info);

    if (bit_depth == 16) png_set_strip_16(png);
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_RGB_ALPHA || color_type == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_strip_alpha(png);

    png_read_update_info(png, info);

    int rowbytes = png_get_rowbytes(png, info);
    unsigned char *image_data = (unsigned char*) malloc(rowbytes * (*height));
    png_bytep *rows = (png_bytep*) malloc(sizeof(png_bytep) * (*height));
    for (int y = 0; y < *height; y++) rows[y] = image_data + y * rowbytes;
    png_read_image(png, rows);

    fclose(fp);
    png_destroy_read_struct(&png, &info, NULL);
    free(rows);

    return image_data;
}

void compress_image(unsigned char *h_pixels, int width, int height) {
    int num_pixels = width * height;
    size_t image_size = num_pixels * CHANNELS * sizeof(unsigned char);

    unsigned char *d_pixels;
    float *d_centroids, *d_sums;
    int *d_labels, *d_counts;
    float *h_centroids = (float*)malloc(K * CHANNELS * sizeof(float));

    cudaMalloc(&d_pixels, image_size);
    cudaMalloc(&d_centroids, K * CHANNELS * sizeof(float));
    cudaMalloc(&d_labels, num_pizels * sizeof(int));
    cudaMalloc(&d_counts, K * sizeof(int));
    cudaMalloc(&d_sums, k * CHANNELS * sizeof(float));
    cudaMemcpy(d_pixels, h_pixels, image_size, cudaMemcpyHostToDevice);
    cudaCheckError();

    initialize_centroids(h_centroids, h_pixels, num_pixels);
    cudaMemcpy(d_centroids, h_centroids, K * CHANNELS * sizeof(float), cudaMemcpyHostToDevice);

    int threads = BLOCK_SIZE;
    int blocks = (num_pixels + threads - 1) / threads;

    // Timing ...
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float total_time = 0.0f;

    for (int itr = 0; itr < MAX_ITER; ++itr) {
        cudaEventRecord(start);

        assign_clusters<<<blocks, threads>>>(d_pixels, d_centroids, d_labels,
                                              num_pixels, d_counts, d_sums);
        cudaDeviceSynchronize();
        
        float h_sums[K * CHANNELS] = {0}; 
        int h_counts[K] = {0};

        cudaMemcpy(h_sums, d_sums, K * CHANNELS * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_counts, d_counts, K * sizeof(int), cudaMemcpyDeviceToHost);

        for (int k = 0; k < K; ++k) {
            if (h_counts[k] > o) {
                for (int c = 0; c < CHANNELS; ++c) {
                    h_centroids[k * CHANNELS + c] = h_sums[k * CHANNELS + c] / h_counts[k];
                }
            }
        }
        cudaMemcpy(d_centroids, h_centroids, K * CHANNELS * sizeof(float), cudaMemcpyHostToDevice);
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        printf("Iteration %d took %.4f ms\n", iter + 1, elapsed);
        total_time += elapsed;
    }

    printf("TOTAL GPU Processing time: %.4f ms\n", total_time);
    quantize_pixels<<<blocks, threads>>>(d_pixels, d_centroids, d_labels, num_pixels);
    cudaMemcpy(h_pixels, d_pixels, image_size, cudaMemcpyDeviceToHost);

    cudaFree(d_pixels);
    cudaFree(d_centroids);
    cudaFree(d_labels);
    cudaFree(d_counts);
    cudaFree(d_sums);
    free(h_centroids);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}