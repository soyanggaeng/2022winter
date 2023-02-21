#include <chrono>
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

__global__ void init(int *a) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  a[i] = i;
}

int main() {
  int *d_a, *a, *a_pinned;
  CHECK_CUDA(cudaMalloc(&d_a, sizeof(int) * (1 << 20)));

  CHECK_CUDA(cudaMallocHost(&a_pinned, sizeof(int) * (1 << 20)));
  a = (int *) malloc(sizeof(int) * (1 << 20));

  for (int i = 0; i < (1 << 20); ++i) { a[i] = i; }
  init<<<(1 << 15), 32>>>(d_a);
  CHECK_CUDA(cudaDeviceSynchronize());

  {
    auto start = std::chrono::system_clock::now();
    // TODO: Run memcpy on pageable memory
    CHECK_CUDA(cudaMemcpy(a, d_a, sizeof(int) * (1<<20), cudaMemcpyDeviceToHost));
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Pageable memory bandwidth: %lf GB/s\n",
           (sizeof(int) / diff.count() / 1000.));
  }
  {
    auto start = std::chrono::system_clock::now();
    // TODO: Run memcpy on pinned memory
    CHECK_CUDA(cudaMemcpy(a_pinned, d_a, sizeof(int) * (1<<20), cudaMemcpyDeviceToHost));
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Pinned memory bandwidth: %lf GB/s\n",
           (sizeof(int) / diff.count() / 1000.));
  }
  return 0;
}
