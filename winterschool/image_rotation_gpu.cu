#include "image_rotation.h"
#include <cstdio>

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

__global__ void rotate_image_kernel(float *input_image, float *output_image, int W, int H, float sin_theta, float cos_theta){
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  int dest_x = blockDim.x * blockIdx.x + threadIdx.x;
  int dest_y = blockDim.y * blockIdx.y + threadIdx.y;

  if (dest_x >= W || dest_y >= H) return;
  
  float xOff = dest_x - x0;
  float yOff = dest_y - y0;
  int src_x = (int) (xOff * cos_theta + yOff * sin_theta + x0);
  int src_y = (int) (yOff * cos_theta - xOff * sin_theta + y0);
  if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
    output_image[dest_y * W + dest_x] = input_image[src_y * W + src_x];
    } else {
        output_image[dest_y * W + dest_x] = 0.0f;
    }
}


float *input_gpu, *output_gpu;
void rotate_image_gpu_initialize(int image_width, int image_height) {
    CHECK_CUDA(cudaMalloc(&input_gpu, image_width*image_height*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output_gpu, image_width*image_height*sizeof(float)));    
}

void rotate_image_gpu(float *input_image, float *output_image, int image_width,
                      int image_height, float sin_theta, float cos_theta) {
                        CHECK_CUDA(cudaMemcpy(input_gpu, input_image, image_width*image_height*sizeof(float), cudaMemcpyHostToDevice));

                        dim3 blockDim(32, 32);
                        dim3 gridDim((image_width+32-1)/32, (image_height+32-1)/32);

                        rotate_image_kernel <<< gridDim, blockDim >>> (input_gpu, output_gpu, image_width, image_height, sin_theta, cos_theta);
                        CHECK_CUDA(cudaDeviceSynchronize());
                        CHECK_CUDA(cudaGetLastError());

                        CHECK_CUDA(cudaMemcpy(output_image, output_gpu, image_width*image_height*sizeof(float), cudaMemcpyDeviceToHost));
                      }

void rotate_image_gpu_finalize() {
    CHECK_CUDA(cudaFree(input_gpu));
    CHECK_CUDA(cudaFree(output_gpu));
}
