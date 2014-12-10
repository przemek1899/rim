
#ifndef MD5_H
#define MD5_H

typedef unsigned char uint1_md5; //  8bit
typedef unsigned int uint4_md5;  // 32bit
void runMD5(unsigned char* result);

__device__ void charMemcpy(uint1_md5 *buffer, uint1_md5 *data, int length);

__device__ uint4_md5 F(uint4_md5 x, uint4_md5 y, uint4_md5 z);
__device__ uint4_md5 G(uint4_md5 x, uint4_md5 y, uint4_md5 z);
__device__ uint4_md5 H(uint4_md5 x, uint4_md5 y, uint4_md5 z);
__device__ uint4_md5 I(uint4_md5 x, uint4_md5 y, uint4_md5 z);

__device__ uint4_md5 rotate_left(uint4_md5 x, int n);

__device__ void FF(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac);
__device__ void GG(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac);
__device__ void HH(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac);
__device__ void II(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac);

__device__ void transform(const uint1_md5 block[64], uint4_md5 state[4]);

__global__ void generateMD5(uint1_md5*);

void printResult(unsigned char* result);

#endif