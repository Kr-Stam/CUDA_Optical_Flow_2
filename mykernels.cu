#include <stdlib.h>
#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void grayscaleAvgGPU(const unsigned char* src, unsigned char* dest, int w, int h){
    int x = threadIdx.x;
    int y = blockIdx.x;

    if(x >= w || y >= h){
        return;
    }
    // printf("%d %d %d", src[pos * 3]);

    int pos = (y * w + x) * 3;
    int avg = (src[pos] + src[pos + 1] + src[pos + 2]) / 3;
    dest[pos] = dest[pos + 1] = dest[pos + 2] = (unsigned char) avg;
}