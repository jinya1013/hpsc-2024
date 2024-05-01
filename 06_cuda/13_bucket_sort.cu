#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void scan(int *a, int *b, int range) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j=1; j<range; j<<=1) {
    b[i] = a[i];
    __syncthreads();
    if (i >= j) a[i] += b[i-j];
    __syncthreads();
  }
}
__global__ void set_bucket(int *bucket, int *key) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(&bucket[key[i]], 1);
}

__global__ void set_offset(int *offset, int *bucket) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= 1) offset[i] = bucket[i-1];
}
__global__ void set_key(int *key, int *offset, int *bucket) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = offset[i];
  for (; bucket[i]>0; bucket[i]--) {
    key[j++] = i;
  }
}

int main() {
  // both n and range should be no more than 1024
  const int n = 50;
  const int range = 128;

  int *key, *bucket, *offset, *tmp;

  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
  cudaMallocManaged(&tmp, range*sizeof(int));

  // init key
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // set bucket
  cudaMemset(bucket, 0, range*sizeof(int));
  set_bucket<<<1, n>>>(bucket, key);
  cudaDeviceSynchronize();

  // set offset
  cudaMemset(offset, 0, range*sizeof(int));
  set_offset<<<1, range>>>(offset, bucket);
  // set tmp
  cudaMemset(tmp, 0, range*sizeof(int));
  cudaDeviceSynchronize();
  scan<<<1, range>>>(offset, tmp, range);
  cudaDeviceSynchronize();

  // set key
  set_key<<<1, range>>>(key, offset, bucket);
  cudaDeviceSynchronize();


  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
  cudaFree(offset);
  cudaFree(tmp);
}
