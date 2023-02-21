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

int main() {
  // TODO
  int count;
  cudaDeviceProp props[8];
  cudaGetDeviceCount(&count);
  printf("Number of devices: %d\n", count);
  for (int d=0; d<count; ++d){
    cudaGetDeviceProperties(&props[d], d);
    printf("\tdevice %d\n", d);
    printf("\t\tname: %s\n", props[0].name);
    printf("\t\tmultiProcessorCount: %d\n", props[0].multiProcessorCount);
    printf("\t\tmaxThreadsPerBlock: %d\n", props[0].maxThreadsPerBlock);
    printf("\t\ttotalGlobalMem: %ld\n", props[0].totalGlobalMem);
    printf("\t\tsharedMemPerBlock: %ld\n", props[0].sharedMemPerBlock);    
  }

  return 0;
}
