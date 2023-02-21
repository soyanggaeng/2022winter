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

__global__ void pythagoras(int *pa, int *pb, int *pc, int *pd) {
  int a = *pa;
  int b = *pb;
  int c = *pc;

  if ((a * a + b * b) == c * c)
    *pd = 1;
  else
    *pd = 0;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <num 1> <num 2> <num 3>\n", argv[0]);
    return 0;
  }

  // TODO
  int *m_a, *m_b, *m_c, *m_d;
  CHECK_CUDA(cudaMalloc(&m_a, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&m_b, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&m_c, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&m_d, sizeof(int)));

  int a = atoi(argv[1]);
  int b = atoi(argv[2]);
  int c = atoi(argv[3]);
  int d;
  CHECK_CUDA(cudaMemcpy(m_a, &a, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(m_b, &b, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(m_c, &c, sizeof(int), cudaMemcpyHostToDevice));
  pythagoras <<< 80, 256 >>> (m_a, m_b, m_c, m_d);
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaMemcpy(&d, m_d, sizeof(int), cudaMemcpyDeviceToHost));
  if (d==1){
    printf("YES\n");
  } else{
    printf("NO\n");
  }

  CHECK_CUDA(cudaFree(m_a));
  CHECK_CUDA(cudaFree(m_b));
  CHECK_CUDA(cudaFree(m_c));
  CHECK_CUDA(cudaFree(m_d));

  return 0;
}
