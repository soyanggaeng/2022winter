#include <cstdio>

__global__ void hello_world(){
    printf("Thread %d: Hello, APWS51!\n", threadIdx.x);
}

int main(){
    hello_world<<<1, 64>>>();
    cudaDeviceSynchronize();
    return 0;
}