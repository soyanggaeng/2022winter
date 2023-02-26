#include <stdlib.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tensor.h"
#include "uNet.h"
#include "util.h"

// Parameters for U-Net
Tensor *inc_double_conv_0_weight;
Tensor *inc_double_conv_1_weight;
Tensor *inc_double_conv_1_bias;
Tensor *inc_double_conv_3_weight;
Tensor *inc_double_conv_4_weight;
Tensor *inc_double_conv_4_bias;
Tensor *down1_maxpool_conv_1_double_conv_0_weight;
Tensor *down1_maxpool_conv_1_double_conv_1_weight;
Tensor *down1_maxpool_conv_1_double_conv_1_bias;
Tensor *down1_maxpool_conv_1_double_conv_3_weight;
Tensor *down1_maxpool_conv_1_double_conv_4_weight;
Tensor *down1_maxpool_conv_1_double_conv_4_bias;
Tensor *down2_maxpool_conv_1_double_conv_0_weight;
Tensor *down2_maxpool_conv_1_double_conv_1_weight;
Tensor *down2_maxpool_conv_1_double_conv_1_bias;
Tensor *down2_maxpool_conv_1_double_conv_3_weight;
Tensor *down2_maxpool_conv_1_double_conv_4_weight;
Tensor *down2_maxpool_conv_1_double_conv_4_bias;
Tensor *up1_up_weight;
Tensor *up1_up_bias;
Tensor *up1_conv_double_conv_0_weight;
Tensor *up1_conv_double_conv_1_weight;
Tensor *up1_conv_double_conv_1_bias;
Tensor *up1_conv_double_conv_3_weight;
Tensor *up1_conv_double_conv_4_weight;
Tensor *up1_conv_double_conv_4_bias;
Tensor *up2_up_weight;
Tensor *up2_up_bias;
Tensor *up2_conv_double_conv_0_weight;
Tensor *up2_conv_double_conv_1_weight;
Tensor *up2_conv_double_conv_1_bias;
Tensor *up2_conv_double_conv_3_weight;
Tensor *up2_conv_double_conv_4_weight;
Tensor *up2_conv_double_conv_4_bias;
Tensor *outc_conv_weight;
Tensor *outc_conv_bias;
Tensor *inc_batchnorm_0_running_mean;
Tensor *inc_batchnorm_0_running_var;
Tensor *down1_batchnorm_0_running_mean;
Tensor *down1_batchnorm_0_running_var;
Tensor *down2_batchnorm_0_running_mean;
Tensor *down2_batchnorm_0_running_var;
Tensor *up1_batchnorm_0_running_mean;
Tensor *up1_batchnorm_0_running_var;
Tensor *up2_batchnorm_0_running_mean;
Tensor *up2_batchnorm_0_running_var;
Tensor *inc_batchnorm_1_running_mean;
Tensor *inc_batchnorm_1_running_var;
Tensor *down1_batchnorm_1_running_mean;
Tensor *down1_batchnorm_1_running_var;
Tensor *down2_batchnorm_1_running_mean;
Tensor *down2_batchnorm_1_running_var;
Tensor *up1_batchnorm_1_running_mean;
Tensor *up1_batchnorm_1_running_var;
Tensor *up2_batchnorm_1_running_mean;
Tensor *up2_batchnorm_1_running_var;

// intermediate features
Tensor *inc_conv_0_output;
Tensor *inc_batchnorm_0_output;
Tensor *inc_conv_1_output;
Tensor *inc_batchnorm_1_output;
Tensor *down1_maxpool2d_0_output;
Tensor *down1_conv_0_output;
Tensor *down1_batchnorm_0_output;
Tensor *down1_conv_1_output;
Tensor *down1_batchnorm_1_output;
Tensor *down2_maxpool2d_0_output;
Tensor *down2_conv_0_output;
Tensor *down2_batchnorm_0_output;
Tensor *down2_conv_1_output;
Tensor *down2_batchnorm_1_output;
Tensor *up1_convt_0_output;
Tensor *up1_concat_0_output;
Tensor *up1_conv_0_output;
Tensor *up1_batchnorm_0_output;
Tensor *up1_conv_1_output;
Tensor *up1_batchnorm_1_output;
Tensor *up2_convt_0_output;
Tensor *up2_concat_0_output;
Tensor *up2_conv_0_output;
Tensor *up2_batchnorm_0_output;
Tensor *up2_conv_1_output;
Tensor *up2_batchnorm_1_output;
Tensor *outc_conv_0_output;

// forward declaration, prototype
void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias);
void ReLU(Tensor *inout);
void BatchNorm2d(Tensor *input, Tensor *gamma, Tensor *beta,
                 Tensor *running_mean, Tensor *running_var, Tensor *output,
                 const float eps, const float momentum);
void ConvTranspose2d(Tensor *input, Tensor *weight, Tensor *bias,
                     Tensor *output, int stride, int pad);
void MaxPool2d(Tensor *input, Tensor *output);
void Concat(Tensor *input1, Tensor *input2, Tensor *output);
void uNet_initialize(int, int, char *);
void uNet(Tensor *, Tensor *);
void uNet_finalize();

/*
 * uNet
 * This model identifies the boundaries of the cars in an image file (input.bin)
 * and removes the background.
 */
void uNet(Tensor *inputN, Tensor *outputN, int N) {
  Tensor *input = new Tensor({1, 3, 128, 191});
  Tensor *output = new Tensor({1, 2, 128, 191});

  // // printf("%d, %d, %d, %d dim of inputN\n", inputN->shape[0], inputN->shape[1], inputN->shape[2], inputN->shape[3]);
  for (int idx = 0; idx < N; ++idx) {
    memcpy(input->buf, inputN->buf + (idx * 1 * 3 * 128 * 191),
      sizeof(float) * 1 * 3 * 128 * 191);
    CHECK_CUDA(cudaMemcpy(input->dev_buf, input->buf, sizeof(float)*input->num_elem(), cudaMemcpyHostToDevice));

    // inc(n_channels, 64)
    Conv2d(input, inc_double_conv_0_weight, NULL, inc_conv_0_output, 1, 1, 1,
           false);
    BatchNorm2d(inc_conv_0_output, inc_double_conv_1_weight,
                inc_double_conv_1_bias, inc_batchnorm_0_running_mean,
                inc_batchnorm_0_running_var, inc_batchnorm_0_output, 1e-5, 0.1);
    ReLU(inc_batchnorm_0_output);
    Conv2d(inc_batchnorm_0_output, inc_double_conv_3_weight, NULL,
           inc_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(inc_conv_1_output, inc_double_conv_4_weight,
                inc_double_conv_4_bias, inc_batchnorm_1_running_mean,
                inc_batchnorm_1_running_var, inc_batchnorm_1_output, 1e-5, 0.1);
    ReLU(inc_batchnorm_1_output);

    // down1(64, 128)
    MaxPool2d(inc_batchnorm_1_output, down1_maxpool2d_0_output);
    Conv2d(down1_maxpool2d_0_output, down1_maxpool_conv_1_double_conv_0_weight,
           NULL, down1_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(down1_conv_0_output, down1_maxpool_conv_1_double_conv_1_weight,
                down1_maxpool_conv_1_double_conv_1_bias,
                down1_batchnorm_0_running_mean, down1_batchnorm_0_running_var,
                down1_batchnorm_0_output, 1e-5, 0.1);
    ReLU(down1_batchnorm_0_output);
    Conv2d(down1_batchnorm_0_output, down1_maxpool_conv_1_double_conv_3_weight,
           NULL, down1_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(down1_conv_1_output, down1_maxpool_conv_1_double_conv_4_weight,
                down1_maxpool_conv_1_double_conv_4_bias,
                down1_batchnorm_1_running_mean, down1_batchnorm_1_running_var,
                down1_batchnorm_1_output, 1e-5, 0.1);
    ReLU(down1_batchnorm_1_output);

    // down2(128, 256)
    MaxPool2d(down1_batchnorm_1_output, down2_maxpool2d_0_output);
    Conv2d(down2_maxpool2d_0_output, down2_maxpool_conv_1_double_conv_0_weight,
           NULL, down2_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(down2_conv_0_output, down2_maxpool_conv_1_double_conv_1_weight,
                down2_maxpool_conv_1_double_conv_1_bias,
                down2_batchnorm_0_running_mean, down2_batchnorm_0_running_var,
                down2_batchnorm_0_output, 1e-5, 0.1);
    ReLU(down2_batchnorm_0_output);
    Conv2d(down2_batchnorm_0_output, down2_maxpool_conv_1_double_conv_3_weight,
           NULL, down2_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(down2_conv_1_output, down2_maxpool_conv_1_double_conv_4_weight,
                down2_maxpool_conv_1_double_conv_4_bias,
                down2_batchnorm_1_running_mean, down2_batchnorm_1_running_var,
                down2_batchnorm_1_output, 1e-5, 0.1);
    ReLU(down2_batchnorm_1_output);

    // up1(256, 128), (up2_concat_0_output, down1_batchnorm_1_output)
    ConvTranspose2d(down2_batchnorm_1_output, up1_up_weight, up1_up_bias,
                    up1_convt_0_output, 2, 0);
    Concat(up1_convt_0_output, down1_batchnorm_1_output, up1_concat_0_output);
    Conv2d(up1_concat_0_output, up1_conv_double_conv_0_weight, NULL,
           up1_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(up1_conv_0_output, up1_conv_double_conv_1_weight,
                up1_conv_double_conv_1_bias, up1_batchnorm_0_running_mean,
                up1_batchnorm_0_running_var, up1_batchnorm_0_output, 1e-5, 0.1);
    ReLU(up1_batchnorm_0_output);
    Conv2d(up1_batchnorm_0_output, up1_conv_double_conv_3_weight, NULL,
           up1_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(up1_conv_1_output, up1_conv_double_conv_4_weight,
                up1_conv_double_conv_4_bias, up1_batchnorm_1_running_mean,
                up1_batchnorm_1_running_var, up1_batchnorm_1_output, 1e-5, 0.1);
    ReLU(up1_batchnorm_1_output);

    // up2(128, 64), (up1_concat_0_output, inc_batchnorm_1_output)
    ConvTranspose2d(up1_batchnorm_1_output, up2_up_weight, up2_up_bias,
                    up2_convt_0_output, 2, 0);
    Concat(up2_convt_0_output, inc_batchnorm_1_output, up2_concat_0_output);
    Conv2d(up2_concat_0_output, up2_conv_double_conv_0_weight, NULL,
           up2_conv_0_output, 1, 1, 1, false);
    BatchNorm2d(up2_conv_0_output, up2_conv_double_conv_1_weight,
                up2_conv_double_conv_1_bias, up2_batchnorm_0_running_mean,
                up2_batchnorm_0_running_var, up2_batchnorm_0_output, 1e-5, 0.1);
    ReLU(up2_batchnorm_0_output);
    Conv2d(up2_batchnorm_0_output, up2_conv_double_conv_3_weight, NULL,
           up2_conv_1_output, 1, 1, 1, false);
    BatchNorm2d(up2_conv_1_output, up2_conv_double_conv_4_weight,
                up2_conv_double_conv_4_bias, up2_batchnorm_1_running_mean,
                up2_batchnorm_1_running_var, up2_batchnorm_1_output, 1e-5, 0.1);
    ReLU(up2_batchnorm_1_output);

    // outc(64, 2)
    Conv2d(up2_batchnorm_1_output, outc_conv_weight, outc_conv_bias, output, 1,
           0, 1, true);

    CHECK_CUDA(cudaMemcpy(outputN->buf + (idx*output->num_elem()), output->dev_buf, sizeof(float)*output->num_elem(), cudaMemcpyDeviceToHost));
  }
}

/* Operations */

/*
 * Convolution
 * input shape = (N, C, H, W)
 * weight shape = (K, C, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 *   where OH = (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
 *         OW = (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1
 */

__global__ void Conv2d_kernel(float *input, float *weight, float *bias, float *output, int stride, int pad, int dilation, bool has_bias, int C, int H, int W, int K, int R, int S, int OH, int OW){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx >= K*OH*OW) return;

  int k = idx / (OH * OW);
  int oh = (idx - k * OH * OW)/OW;
  int ow = (idx - k * OH * OW) - oh * OW;

  float o = has_bias ? bias[k] : 0;
  #pragma unroll 8
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        int h = oh * stride - pad + r * dilation;
        int w = ow * stride - pad + s * dilation;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        float i = input[c * H * W + h * W + w];
        float f = weight[k * C * R * S + c * R * S + r * S + s];
        o += i * f;
      }
    }
  }
  output[k * OH * OW + oh * OW + ow] = o;
}

void Conv2d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int pad, int dilation, bool has_bias) {
  double start = get_time();

  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[0], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OH == (H + 2 * pad - dilation * (R - 1) - 1) / stride + 1,
              "[Conv2d] Output height mismatch");
  CHECK_ERROR(OW == (W + 2 * pad - dilation * (S - 1) - 1) / stride + 1,
              "[Conv2d] Output width mismatch");
  CHECK_ERROR(weight->shape[1] == C && (!has_bias || bias->shape[0] == K) &&
                  output->shape[1] == K,
              "[Conv2d] Channel size mismatch");

#ifdef TEST
#pragma omp parallel for
#endif

  dim3 blockDim(32, 32);
  dim3 gridDim((K*OH*OW+32-1)/32, 1);

  if (has_bias==true) Conv2d_kernel <<< gridDim, blockDim >>> (input->dev_buf, weight->dev_buf, bias->dev_buf, output->dev_buf, stride, pad, dilation, has_bias, C, H, W, K, R, S, OH, OW);
  else Conv2d_kernel <<< gridDim, blockDim >>> (input->dev_buf, weight->dev_buf, NULL, output->dev_buf, stride, pad, dilation, has_bias, C, H, W, K, R, S, OH, OW);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());
  
  double end = get_time();
  // printf("%f seconds elapsed in Conv2d\n", end - start);
}

/*
 * ReLU
 * input shape = (N, C, H, W)
 * output shape = (N, C, H, W)
 * Formula: y = max(x, 0)
 */
__global__ void ReLU_kernel(float *inout_buf, int C, int H, int W){
  int idx = blockDim.x * blockIdx.x + threadIdx.x;

// int idx = c*H*W+h*W+w;
  if (idx >= C*H*W) return;

  int c = idx / (H*W);
  int h = (idx - c * H * W)/W;
  int w = (idx - c * H * W) - h * W;

  inout_buf[c*H*W+h*W+w] = inout_buf[c*H*W+h*W+w] > 0 ? inout_buf[c*H*W+h*W+w] : 0;
}

void ReLU(Tensor *inout) {
  double start = get_time();
  int C = inout->shape[1], H = inout->shape[2], W = inout->shape[3];

  dim3 blockDim(32, 32);
  dim3 gridDim((C*H*W+32-1)/32, 1);
  ReLU_kernel <<< gridDim, blockDim >>> (inout->dev_buf, C, H, W);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  double end = get_time();
  // printf("%f seconds elapsed in ReLU\n", end - start);
}

/*
 * Batch Normaliztion
 * input shape = (N, C, H, W)
 * gamma shape = (C)
 * beta shape = (C)
 * output shape = (N, C, H, W)
 */

 __global__ void BatchNorm2d_kernel(float *input, float *gamma, float *beta, float *running_mean, float *running_var, float *output, const float eps, const float momentum, int N, int C, int H, int W){
  int h = blockDim.x * blockIdx.x + threadIdx.x;
  int w = blockDim.y * blockIdx.y + threadIdx.y;

  #pragma unroll 4
  for (int c = 0; c < C; ++c) {
    for (int n = 0; n < N; ++n) {
      float mean = running_mean[c];
      float variance = running_var[c];
      float x = input[n * C * H * W + c * H * W + h * W + w];
      float x_hat = (x - mean) / sqrt(variance + eps);
      output[n * C * H * W + c * H * W + h * W + w] =
          gamma[c] * x_hat + beta[c];
    }
  }
 }

void BatchNorm2d(Tensor *input, Tensor *gamma, Tensor *beta,
                 Tensor *running_mean, Tensor *running_var, Tensor *output,
                 const float eps, const float momentum) {
  double start = get_time();
  int N = input->shape[0], C = input->shape[1], H = input->shape[2],
      W = input->shape[3];

  CHECK_ERROR(gamma->shape[0] == C && beta->shape[0] == C,
              "[BatchNorm2d] gamma, beta shape mismatch");
  CHECK_ERROR(
      output->shape[1] == C && output->shape[2] == H && output->shape[3] == W,
      "[BatchNorm2d] Output shape mismatch");

  dim3 blockDim(32, 32);
  dim3 gridDim((H+32-1)/32, (W+32-1)/32);
  BatchNorm2d_kernel <<< gridDim, blockDim >>> (input->dev_buf, gamma->dev_buf, beta->dev_buf, running_mean->dev_buf, running_var->dev_buf, output->dev_buf, eps, momentum, N, C, H, W);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  double end = get_time();
  // printf("%f seconds elapsed in BatchNorm2d\n", end - start);
}

/*
 * Transposed convolution
 * input shape = (N, C, H, W)
 * weight shape = (C, K, R, S)
 * bias shape = (K)
 * output shape = (N, K, OH, OW)
 *   where OH = (H - 1) * stride - 2 * pad + R
 *         OW = (W - 1) * stride - 2 * pad + S
 */
__global__ void ConvTranspose2d_kernel(float *input, float *weight, float *bias, float *output, int stride, int pad, int C, int H, int W, int K, int R, int S, int OH, int OW){
  int oh = blockDim.x * blockIdx.x + threadIdx.x;
  int ow = blockDim.y * blockIdx.y + threadIdx.y;
  
    for (int k = 0; k < K; ++k) {
      float o = bias[k];
        for (int c = 0; c < C; ++c) {
          for (int r = 0; r < R; ++r) {
            for (int s = 0; s < S; ++s) {
              if ((oh + pad - r) % stride != 0) continue;
              if ((ow + pad - s) % stride != 0) continue;
              int h = (oh + pad - r) / stride;
              int w = (ow + pad - s) / stride;
              if (h < 0 || h >= H || w < 0 || w >= W) continue;
              float i = input[c * H * W + h * W + w];
              float f = weight[c * K * R * S + k * R * S + r * S + s];
              o += i * f;
            }
          }
        }
        output[k * OH * OW + oh * OW + ow] = o;
      }
    }

void ConvTranspose2d(Tensor *input, Tensor *weight, Tensor *bias,
                     Tensor *output, int stride, int pad) {
  double start = get_time();
  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int K = weight->shape[1], R = weight->shape[2], S = weight->shape[3];
  int OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OH == (H - 1) * stride - 2 * pad + R,
              "[ConvT2d] Output height mismatch");
  CHECK_ERROR(OW == (W - 1) * stride - 2 * pad + S,
              "[ConvT2d] Output width mismatch");
  CHECK_ERROR(
      weight->shape[0] == C && bias->shape[0] == K && output->shape[1] == K,
      "[ConvT2d] Channel size mismatch");

  dim3 blockDim(32, 32);
  dim3 gridDim((OH+32-1)/32, (OW+32-1)/32);
  ConvTranspose2d_kernel <<< gridDim, blockDim >>> (input->dev_buf, weight->dev_buf, bias->dev_buf, output->dev_buf, stride, pad, C, H, W, K, R, S, OH, OW);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  double end = get_time();
  // printf("%f seconds elapsed in ConvTranspose2d\n", end - start);
}

__device__ float max4(float in0, float in1, float in2, float in3) {
  float max = in0;

  if (in1 > max) max = in1;
  if (in2 > max) max = in2;
  if (in3 > max) max = in3;
  return max;
}

/*
 * MaxPool2d
 * input shape = (N, C, H, W)
 * output shape = (N, OC, OH, OW)
 *   where OH = H / 2
 *         OW = W / 2
 */
 __global__ void MaxPool2d_kernel(float *input, float *output, int C, int H, int W, int OC, int OH, int OW){
  int oh = blockDim.x * blockIdx.x + threadIdx.x;
  int ow = blockDim.y * blockIdx.y + threadIdx.y;

  for (int oc = 0; oc < OC; ++oc) {
    float in0 = input[oc * H * W + 2 * oh * W + 2 * ow];
    float in1 = input[oc * H * W + 2 * oh * W + 2 * ow + 1];
    float in2 = input[oc * H * W + (2 * oh + 1) * W + 2 * ow];
    float in3 = input[oc * H * W + (2 * oh + 1) * W + 2 * ow + 1];
    output[oc * OH * OW + oh * OW + ow] = max4(in0, in1, in2, in3);
  }
}

void MaxPool2d(Tensor *input, Tensor *output) {
  double start = get_time();

  int C = input->shape[1], H = input->shape[2], W = input->shape[3];
  int OC = output->shape[1], OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OW == W / 2, "[MaxPool2d] Output width mismatch");
  CHECK_ERROR(OH == H / 2, "[MaxPool2d] Output height mismatch");
  CHECK_ERROR(OC == C, "[MaxPool2d] Output channel mismatch");

  dim3 blockDim(32, 32);
  dim3 gridDim((OH+32-1)/32, (OW+32-1)/32);

  MaxPool2d_kernel <<< gridDim, blockDim >>> (input->dev_buf, output->dev_buf, C, H, W, OC, OH, OW);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());  

  double end = get_time();
  // printf("%f seconds elapsed in MaxPool2d\n", end - start);
}

/*
 * Concat
 * input1 shape = (N, C1, H1, W1)
 * input2 shape = (N, C2, H2, W2)
 * output shape = (N, OC, OH, OW)
 *   where OH = H2, H1
 *         OW = W2 = W1 + 1
 */

 __global__ void Concat_kernel(float *input1, float *input2, float *output, int C1, int H1, int W1, int C2, int H2, int W2, int OC, int OH, int OW){
  int oh = blockDim.x * blockIdx.x + threadIdx.x;
  int ow = blockDim.y * blockIdx.y + threadIdx.y;

  for (int oc = 0; oc < OC / 2; ++oc) {
    output[oc * OH * OW + oh * OW + ow] =
      input2[oc * OH * OW + oh * OW + ow];
  }

  for (int oc = OC / 2; oc < OC; ++oc) {
    if (ow == OW - 1)
      output[oc * OH * OW + oh * OW + ow] = 0.0;  // zero padding
    else
      output[oc * OH * OW + oh * OW + ow] =
        input1[(oc - OC / 2) * H1 * W1 + oh * W1 + ow];
  }
}

void Concat(Tensor *input1, Tensor *input2, Tensor *output) {
  double start = get_time();
  int C1 = input1->shape[1], H1 = input1->shape[2], W1 = input1->shape[3];
  int C2 = input2->shape[1], H2 = input2->shape[2], W2 = input2->shape[3];
  int OC = output->shape[1], OH = output->shape[2], OW = output->shape[3];

  CHECK_ERROR(OC == C1 * 2 && OC == C2 * 2, "[Concat] Output channel mismatch");
  CHECK_ERROR(OW == W1 + 1 && OW == W2, "[Concat] Output width mismatch");
  CHECK_ERROR(OH == H1 && OH == H2, "[Concat] Output height mismatch");

  dim3 blockDim(32, 32);
  dim3 gridDim((OH+32-1)/32, (OW+32-1)/32);

  Concat_kernel <<< gridDim, blockDim >>> (input1->dev_buf, input2->dev_buf, output->dev_buf, C1, H1, W1, C2, H2, W2, OC, OH, OW);

  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaGetLastError());

  double end = get_time();
  // printf("%f seconds elapsed in Concat\n", end - start);
}

/*
 * uNet_initialize
 * Initialize the model. Do input-independent job here.
 */
void uNet_initialize(int N, char *parameter_fname) {
  double start = get_time();
  size_t parameter_binary_size = 0;
  float *parameter =
      (float *) read_binary(parameter_fname, &parameter_binary_size);

  // Parameters
  inc_double_conv_0_weight = new Tensor({64, 3, 3, 3}, parameter + OFFSET0);
  inc_double_conv_1_weight = new Tensor({64}, parameter + OFFSET1);
  inc_double_conv_1_bias = new Tensor({64}, parameter + OFFSET2);
  inc_double_conv_3_weight = new Tensor({64, 64, 3, 3}, parameter + OFFSET3);
  inc_double_conv_4_weight = new Tensor({64}, parameter + OFFSET4);
  inc_double_conv_4_bias = new Tensor({64}, parameter + OFFSET5);
  down1_maxpool_conv_1_double_conv_0_weight =
      new Tensor({128, 64, 3, 3}, parameter + OFFSET6);
  down1_maxpool_conv_1_double_conv_1_weight =
      new Tensor({128}, parameter + OFFSET7);
  down1_maxpool_conv_1_double_conv_1_bias =
      new Tensor({128}, parameter + OFFSET8);
  down1_maxpool_conv_1_double_conv_3_weight =
      new Tensor({128, 128, 3, 3}, parameter + OFFSET9);
  down1_maxpool_conv_1_double_conv_4_weight =
      new Tensor({128}, parameter + OFFSET10);
  down1_maxpool_conv_1_double_conv_4_bias =
      new Tensor({128}, parameter + OFFSET11);
  down2_maxpool_conv_1_double_conv_0_weight =
      new Tensor({256, 128, 3, 3}, parameter + OFFSET12);
  down2_maxpool_conv_1_double_conv_1_weight =
      new Tensor({256}, parameter + OFFSET13);
  down2_maxpool_conv_1_double_conv_1_bias =
      new Tensor({256}, parameter + OFFSET14);
  down2_maxpool_conv_1_double_conv_3_weight =
      new Tensor({256, 256, 3, 3}, parameter + OFFSET15);
  down2_maxpool_conv_1_double_conv_4_weight =
      new Tensor({256}, parameter + OFFSET16);
  down2_maxpool_conv_1_double_conv_4_bias =
      new Tensor({256}, parameter + OFFSET17);
  up1_up_weight = new Tensor({256, 128, 2, 2}, parameter + OFFSET18);
  up1_up_bias = new Tensor({128}, parameter + OFFSET19);
  up1_conv_double_conv_0_weight =
      new Tensor({128, 256, 3, 3}, parameter + OFFSET20);
  up1_conv_double_conv_1_weight = new Tensor({128}, parameter + OFFSET21);
  up1_conv_double_conv_1_bias = new Tensor({128}, parameter + OFFSET22);
  up1_conv_double_conv_3_weight =
      new Tensor({128, 128, 3, 3}, parameter + OFFSET23);
  up1_conv_double_conv_4_weight = new Tensor({128}, parameter + OFFSET24);
  up1_conv_double_conv_4_bias = new Tensor({128}, parameter + OFFSET25);
  up2_up_weight = new Tensor({128, 64, 2, 2}, parameter + OFFSET26);
  up2_up_bias = new Tensor({64}, parameter + OFFSET27);
  up2_conv_double_conv_0_weight =
      new Tensor({64, 128, 3, 3}, parameter + OFFSET28);
  up2_conv_double_conv_1_weight = new Tensor({64}, parameter + OFFSET29);
  up2_conv_double_conv_1_bias = new Tensor({64}, parameter + OFFSET30);
  up2_conv_double_conv_3_weight =
      new Tensor({64, 64, 3, 3}, parameter + OFFSET31);
  up2_conv_double_conv_4_weight = new Tensor({64}, parameter + OFFSET32);
  up2_conv_double_conv_4_bias = new Tensor({64}, parameter + OFFSET33);
  outc_conv_weight = new Tensor({2, 64, 1, 1}, parameter + OFFSET34);
  outc_conv_bias = new Tensor({2}, parameter + OFFSET35);
  inc_batchnorm_0_running_mean = new Tensor({64}, parameter + OFFSET36);
  inc_batchnorm_0_running_var = new Tensor({64}, parameter + OFFSET37);
  inc_batchnorm_1_running_mean = new Tensor({64}, parameter + OFFSET38);
  inc_batchnorm_1_running_var = new Tensor({64}, parameter + OFFSET39);
  down1_batchnorm_0_running_mean = new Tensor({128}, parameter + OFFSET40);
  down1_batchnorm_0_running_var = new Tensor({128}, parameter + OFFSET41);
  down1_batchnorm_1_running_mean = new Tensor({128}, parameter + OFFSET42);
  down1_batchnorm_1_running_var = new Tensor({128}, parameter + OFFSET43);
  down2_batchnorm_0_running_mean = new Tensor({256}, parameter + OFFSET44);
  down2_batchnorm_0_running_var = new Tensor({256}, parameter + OFFSET45);
  down2_batchnorm_1_running_mean = new Tensor({256}, parameter + OFFSET46);
  down2_batchnorm_1_running_var = new Tensor({256}, parameter + OFFSET47);
  up1_batchnorm_0_running_mean = new Tensor({128}, parameter + OFFSET48);
  up1_batchnorm_0_running_var = new Tensor({128}, parameter + OFFSET49);
  up1_batchnorm_1_running_mean = new Tensor({128}, parameter + OFFSET50);
  up1_batchnorm_1_running_var = new Tensor({128}, parameter + OFFSET51);
  up2_batchnorm_0_running_mean = new Tensor({64}, parameter + OFFSET52);
  up2_batchnorm_0_running_var = new Tensor({64}, parameter + OFFSET53);
  up2_batchnorm_1_running_mean = new Tensor({64}, parameter + OFFSET54);
  up2_batchnorm_1_running_var = new Tensor({64}, parameter + OFFSET55);

  // Activations
  inc_conv_0_output = new Tensor({1, 64, 128, 191});
  inc_batchnorm_0_output = new Tensor({1, 64, 128, 191});
  inc_conv_1_output = new Tensor({1, 64, 128, 191});
  inc_batchnorm_1_output = new Tensor({1, 64, 128, 191});

  down1_maxpool2d_0_output = new Tensor({1, 64, 64, 95});
  down1_conv_0_output = new Tensor({1, 128, 64, 95});
  down1_batchnorm_0_output = new Tensor({1, 128, 64, 95});
  down1_conv_1_output = new Tensor({1, 128, 64, 95});
  down1_batchnorm_1_output = new Tensor({1, 128, 64, 95});

  down2_maxpool2d_0_output = new Tensor({1, 128, 32, 47});
  down2_conv_0_output = new Tensor({1, 256, 32, 47});
  down2_batchnorm_0_output = new Tensor({1, 256, 32, 47});
  down2_conv_1_output = new Tensor({1, 256, 32, 47});
  down2_batchnorm_1_output = new Tensor({1, 256, 32, 47});

  up1_convt_0_output = new Tensor({1, 128, 64, 94});
  up1_concat_0_output = new Tensor({1, 256, 64, 95});
  up1_conv_0_output = new Tensor({1, 128, 64, 95});
  up1_batchnorm_0_output = new Tensor({1, 128, 64, 95});
  up1_conv_1_output = new Tensor({1, 128, 64, 95});
  up1_batchnorm_1_output = new Tensor({1, 128, 64, 95});

  up2_convt_0_output = new Tensor({1, 64, 128, 190});
  up2_concat_0_output = new Tensor({1, 128, 128, 191});
  up2_conv_0_output = new Tensor({1, 64, 128, 191});
  up2_batchnorm_0_output = new Tensor({1, 64, 128, 191});
  up2_conv_1_output = new Tensor({1, 64, 128, 191});
  up2_batchnorm_1_output = new Tensor({1, 64, 128, 191});
  outc_conv_0_output = new Tensor({1, 2, 128, 191});

  double end = get_time();
  // printf("%f seconds elapsed in uNet initialize\n", end - start);

}

/*
 * uNet_finalize
 * Finalize the model.
 */
void uNet_finalize() {
  // delete parameters
  delete inc_double_conv_0_weight;
  delete inc_double_conv_1_weight;
  delete inc_double_conv_1_bias;
  delete inc_double_conv_3_weight;
  delete inc_double_conv_4_weight;
  delete inc_double_conv_4_bias;
  delete down1_maxpool_conv_1_double_conv_0_weight;
  delete down1_maxpool_conv_1_double_conv_1_weight;
  delete down1_maxpool_conv_1_double_conv_1_bias;
  delete down1_maxpool_conv_1_double_conv_3_weight;
  delete down1_maxpool_conv_1_double_conv_4_weight;
  delete down1_maxpool_conv_1_double_conv_4_bias;
  delete down2_maxpool_conv_1_double_conv_0_weight;
  delete down2_maxpool_conv_1_double_conv_1_weight;
  delete down2_maxpool_conv_1_double_conv_1_bias;
  delete down2_maxpool_conv_1_double_conv_3_weight;
  delete down2_maxpool_conv_1_double_conv_4_weight;
  delete down2_maxpool_conv_1_double_conv_4_bias;
  delete up1_up_weight;
  delete up1_up_bias;
  delete up1_conv_double_conv_0_weight;
  delete up1_conv_double_conv_1_weight;
  delete up1_conv_double_conv_1_bias;
  delete up1_conv_double_conv_3_weight;
  delete up1_conv_double_conv_4_weight;
  delete up1_conv_double_conv_4_bias;
  delete up2_up_weight;
  delete up2_up_bias;
  delete up2_conv_double_conv_0_weight;
  delete up2_conv_double_conv_1_weight;
  delete up2_conv_double_conv_1_bias;
  delete up2_conv_double_conv_3_weight;
  delete up2_conv_double_conv_4_weight;
  delete up2_conv_double_conv_4_bias;
  delete outc_conv_weight;
  delete outc_conv_bias;
  delete inc_batchnorm_0_running_mean;
  delete inc_batchnorm_0_running_var;
  delete down1_batchnorm_0_running_mean;
  delete down1_batchnorm_0_running_var;
  delete down2_batchnorm_0_running_mean;
  delete down2_batchnorm_0_running_var;
  delete up1_batchnorm_0_running_mean;
  delete up1_batchnorm_0_running_var;
  delete up2_batchnorm_0_running_mean;
  delete up2_batchnorm_0_running_var;
  delete inc_batchnorm_1_running_mean;
  delete inc_batchnorm_1_running_var;
  delete down1_batchnorm_1_running_mean;
  delete down1_batchnorm_1_running_var;
  delete down2_batchnorm_1_running_mean;
  delete down2_batchnorm_1_running_var;
  delete up1_batchnorm_1_running_mean;
  delete up1_batchnorm_1_running_var;
  delete up2_batchnorm_1_running_mean;
  delete up2_batchnorm_1_running_var;

  // delete activations
  delete inc_conv_0_output;
  delete inc_batchnorm_0_output;
  delete inc_conv_1_output;
  delete inc_batchnorm_1_output;
  delete down1_maxpool2d_0_output;
  delete down1_conv_0_output;
  delete down1_batchnorm_0_output;
  delete down1_conv_1_output;
  delete down1_batchnorm_1_output;
  delete down2_maxpool2d_0_output;
  delete down2_conv_0_output;
  delete down2_batchnorm_0_output;
  delete down2_conv_1_output;
  delete down2_batchnorm_1_output;
  delete up1_convt_0_output;
  delete up1_concat_0_output;
  delete up1_conv_0_output;
  delete up1_batchnorm_0_output;
  delete up1_conv_1_output;
  delete up1_batchnorm_1_output;
  delete up2_convt_0_output;
  delete up2_concat_0_output;
  delete up2_conv_0_output;
  delete up2_batchnorm_0_output;
  delete up2_conv_1_output;
  delete up2_batchnorm_1_output;
  delete outc_conv_0_output;
}
