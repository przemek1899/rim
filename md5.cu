/* MD5
 converted to C++ class by Frank Thilo (thilo@unix-ag.org)
 for bzflag (http://www.bzflag.org)
 
   based on:
 
   md5.h and md5.c
   reference implemantion of RFC 1321
 
   Copyright (C) 1991-2, RSA Data Security, Inc. Created 1991. All
rights reserved.
 
License to copy and use this software is granted provided that it
is identified as the "RSA Data Security, Inc. MD5 Message-Digest
Algorithm" in all material mentioning or referencing this software
or this function.
 
License is also granted to make and use derivative works provided
that such works are identified as "derived from the RSA Data
Security, Inc. MD5 Message-Digest Algorithm" in all material
mentioning or referencing the derived work.
 
RSA Data Security, Inc. makes no representations concerning either
the merchantability of this software or the suitability of this
software for any particular purpose. It is provided "as is"
without express or implied warranty of any kind.
 
These notices must be retained in any copies of any part of this
documentation and/or software.
 
*/

#include "cuda_runtime.h"
#include "md5.cuh"
#include "helper_cuda.h"
#include <stdio.h>

#define S11 7
#define S12 12
#define S13 17
#define S14 22
#define S21 5
#define S22 9
#define S23 14
#define S24 20
#define S31 4
#define S32 11
#define S33 16
#define S34 23
#define S41 6
#define S42 10
#define S43 15
#define S44 21

uint1_md5 *foundCollision; // the result
char *testPointer;

//declare constant memory
__constant__ unsigned char padding[64]; 
unsigned char * message;

__device__ void charMemcpy(uint1_md5 *buffer, uint1_md5 *data, int length){

	int i;
	for(i=0; i<length; i++){
		buffer[i] = data[i];
	}
}

unsigned char hostPadding[64] = {
    0x80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

void runMD5(unsigned char* result, unsigned char * message_from_host, int length){

	checkCudaErrors(cudaSetDevice(0));
	//kopiowanie do pamieci stalej urzadzenia
	checkCudaErrors(cudaMemcpyToSymbol(padding, &hostPadding, sizeof(char)*64, 0, cudaMemcpyHostToDevice));

	//checkCudaErrors(cudaMalloc((void**)&testPointer, 16*sizeof(char)));
	checkCudaErrors(cudaMalloc((void**)&foundCollision, 16*sizeof(char)));
	checkCudaErrors(cudaMalloc((void**)&message, length*sizeof(char)));

	checkCudaErrors(cudaMemcpy(message, message_from_host, length*sizeof(char), cudaMemcpyHostToDevice));
	
	generateMD5<<<1, 1>>>(foundCollision, message, length);
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(result, foundCollision, 16*sizeof(char), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(foundCollision));
	checkCudaErrors(cudaFree(message));

	//resetowanie urz�dzenia
	checkCudaErrors(cudaDeviceReset());
    return;
}

void printResult(unsigned char* result){
	char buf[33];
	for (int i=0; i<16; i++)
		sprintf(buf+i*2, "%02x", result[i]);
	buf[32]=0;
	printf("%s\n", buf);
}

 
// F, G, H and I are basic MD5 functions.
__device__ uint4_md5 F(uint4_md5 x, uint4_md5 y, uint4_md5 z) {
  return x&y | ~x&z;
}
 
__device__ uint4_md5 G(uint4_md5 x, uint4_md5 y, uint4_md5 z) {
  return x&z | y&~z;
}
 
__device__ uint4_md5 H(uint4_md5 x, uint4_md5 y, uint4_md5 z) {
  return x^y^z;
}
 
__device__ uint4_md5 I(uint4_md5 x, uint4_md5 y, uint4_md5 z) {
  return y ^ (x | ~z);
}
 
// rotate_left rotates x left n bits.
__device__ uint4_md5 rotate_left(uint4_md5 x, int n) {
  return (x << n) | (x >> (32-n));
}

// FF, GG, HH, and II transformations for rounds 1, 2, 3, and 4.
// Rotation is separate from addition to prevent recomputation.
__device__ void FF(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac) {
  a = rotate_left(a+ F(b,c,d) + x + ac, s) + b;
}
 
__device__ void GG(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac) {
  a = rotate_left(a + G(b,c,d) + x + ac, s) + b;
}
 
__device__ void HH(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac) {
  a = rotate_left(a + H(b,c,d) + x + ac, s) + b;
}
 
__device__ void II(uint4_md5 &a, uint4_md5 b, uint4_md5 c, uint4_md5 d, uint4_md5 x, uint4_md5 s, uint4_md5 ac) {
  a = rotate_left(a + I(b,c,d) + x + ac, s) + b;
}


__device__ void transform(const uint1_md5 block[64], uint4_md5 state[4]){
	uint4_md5 a = state[0], b = state[1], c = state[2], d = state[3], x[16];
  //decode (x, block, 64);
  //ponizszy kod zastepuje powyzsza funcke decode
	for (unsigned int i = 0, j = 0; j < 64; i++, j += 4)
    x[i] = ((uint4_md5)block[j]) | (((uint4_md5)block[j+1]) << 8) |
      (((uint4_md5)block[j+2]) << 16) | (((uint4_md5)block[j+3]) << 24);
  //------------koniec decode ------------------------------
 
  /* Round 1 */
  FF (a, b, c, d, x[ 0], S11, 0xd76aa478); /* 1 */
  FF (d, a, b, c, x[ 1], S12, 0xe8c7b756); /* 2 */
  FF (c, d, a, b, x[ 2], S13, 0x242070db); /* 3 */
  FF (b, c, d, a, x[ 3], S14, 0xc1bdceee); /* 4 */
  FF (a, b, c, d, x[ 4], S11, 0xf57c0faf); /* 5 */
  FF (d, a, b, c, x[ 5], S12, 0x4787c62a); /* 6 */
  FF (c, d, a, b, x[ 6], S13, 0xa8304613); /* 7 */
  FF (b, c, d, a, x[ 7], S14, 0xfd469501); /* 8 */
  FF (a, b, c, d, x[ 8], S11, 0x698098d8); /* 9 */
  FF (d, a, b, c, x[ 9], S12, 0x8b44f7af); /* 10 */
  FF (c, d, a, b, x[10], S13, 0xffff5bb1); /* 11 */
  FF (b, c, d, a, x[11], S14, 0x895cd7be); /* 12 */
  FF (a, b, c, d, x[12], S11, 0x6b901122); /* 13 */
  FF (d, a, b, c, x[13], S12, 0xfd987193); /* 14 */
  FF (c, d, a, b, x[14], S13, 0xa679438e); /* 15 */
  FF (b, c, d, a, x[15], S14, 0x49b40821); /* 16 */
 
  /* Round 2 */
  GG (a, b, c, d, x[ 1], S21, 0xf61e2562); /* 17 */
  GG (d, a, b, c, x[ 6], S22, 0xc040b340); /* 18 */
  GG (c, d, a, b, x[11], S23, 0x265e5a51); /* 19 */
  GG (b, c, d, a, x[ 0], S24, 0xe9b6c7aa); /* 20 */
  GG (a, b, c, d, x[ 5], S21, 0xd62f105d); /* 21 */
  GG (d, a, b, c, x[10], S22,  0x2441453); /* 22 */
  GG (c, d, a, b, x[15], S23, 0xd8a1e681); /* 23 */
  GG (b, c, d, a, x[ 4], S24, 0xe7d3fbc8); /* 24 */
  GG (a, b, c, d, x[ 9], S21, 0x21e1cde6); /* 25 */
  GG (d, a, b, c, x[14], S22, 0xc33707d6); /* 26 */
  GG (c, d, a, b, x[ 3], S23, 0xf4d50d87); /* 27 */
  GG (b, c, d, a, x[ 8], S24, 0x455a14ed); /* 28 */
  GG (a, b, c, d, x[13], S21, 0xa9e3e905); /* 29 */
  GG (d, a, b, c, x[ 2], S22, 0xfcefa3f8); /* 30 */
  GG (c, d, a, b, x[ 7], S23, 0x676f02d9); /* 31 */
  GG (b, c, d, a, x[12], S24, 0x8d2a4c8a); /* 32 */
 
  /* Round 3 */
  HH (a, b, c, d, x[ 5], S31, 0xfffa3942); /* 33 */
  HH (d, a, b, c, x[ 8], S32, 0x8771f681); /* 34 */
  HH (c, d, a, b, x[11], S33, 0x6d9d6122); /* 35 */
  HH (b, c, d, a, x[14], S34, 0xfde5380c); /* 36 */
  HH (a, b, c, d, x[ 1], S31, 0xa4beea44); /* 37 */
  HH (d, a, b, c, x[ 4], S32, 0x4bdecfa9); /* 38 */
  HH (c, d, a, b, x[ 7], S33, 0xf6bb4b60); /* 39 */
  HH (b, c, d, a, x[10], S34, 0xbebfbc70); /* 40 */
  HH (a, b, c, d, x[13], S31, 0x289b7ec6); /* 41 */
  HH (d, a, b, c, x[ 0], S32, 0xeaa127fa); /* 42 */
  HH (c, d, a, b, x[ 3], S33, 0xd4ef3085); /* 43 */
  HH (b, c, d, a, x[ 6], S34,  0x4881d05); /* 44 */
  HH (a, b, c, d, x[ 9], S31, 0xd9d4d039); /* 45 */
  HH (d, a, b, c, x[12], S32, 0xe6db99e5); /* 46 */
  HH (c, d, a, b, x[15], S33, 0x1fa27cf8); /* 47 */
  HH (b, c, d, a, x[ 2], S34, 0xc4ac5665); /* 48 */
 
  /* Round 4 */
  II (a, b, c, d, x[ 0], S41, 0xf4292244); /* 49 */
  II (d, a, b, c, x[ 7], S42, 0x432aff97); /* 50 */
  II (c, d, a, b, x[14], S43, 0xab9423a7); /* 51 */
  II (b, c, d, a, x[ 5], S44, 0xfc93a039); /* 52 */
  II (a, b, c, d, x[12], S41, 0x655b59c3); /* 53 */
  II (d, a, b, c, x[ 3], S42, 0x8f0ccc92); /* 54 */
  II (c, d, a, b, x[10], S43, 0xffeff47d); /* 55 */
  II (b, c, d, a, x[ 1], S44, 0x85845dd1); /* 56 */
  II (a, b, c, d, x[ 8], S41, 0x6fa87e4f); /* 57 */
  II (d, a, b, c, x[15], S42, 0xfe2ce6e0); /* 58 */
  II (c, d, a, b, x[ 6], S43, 0xa3014314); /* 59 */
  II (b, c, d, a, x[13], S44, 0x4e0811a1); /* 60 */
  II (a, b, c, d, x[ 4], S41, 0xf7537e82); /* 61 */
  II (d, a, b, c, x[11], S42, 0xbd3af235); /* 62 */
  II (c, d, a, b, x[ 2], S43, 0x2ad7d2bb); /* 63 */
  II (b, c, d, a, x[ 9], S44, 0xeb86d391); /* 64 */
 
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
 
  // Zeroize sensitive information.
  // memset(x, 0, sizeof x);
}

__global__ void generateMD5(uint1_md5* foundCollision, unsigned char * message, int message_length){

	uint1_md5 buffer[64]; //bytes that didn't fit in the last chunk

	//uint1_md5 message[11] = {'a','l','a',' ', 'm', 'a',' ','k','o','t','a'};
	uint4_md5 length = message_length; // na razie zafiksowane, bo chyba nie bedziemy obliczac w gpu dlugosci stringa
	// init -----------------------------------------------------
	// finalize = false;
	uint4_md5 count[2];   // 64bit counter for number of bits (lo, hi)
	count[0] = 0;
	count[1] = 0;

	uint4_md5 state[4];
	// load magic initialization constants.
	state[0] = 0x67452301;
	state[1] = 0xefcdab89;
	state[2] = 0x98badcfe;
	state[3] = 0x10325476;

	uint1_md5 digest[16]; // the result

	// update ----------------------------------------------------
	// compute number of bytes mod 64
	uint4_md5 index = count[0] / 8 % 64;

	// Update number of bits
	if ((count[0] += (length << 3)) < (length << 3))
		count[1]++;
	count[1] += (length >> 29);

	// number of bytes we need to fill in buffer
  uint4_md5 firstpart = 64 - index;
 
  uint4_md5 i=0;
 
  // transform as many times as possible.
  if (length >= firstpart)
  {
    // fill buffer first, transform
    charMemcpy(&buffer[index], message, firstpart);
    transform(buffer, state);
 
    // transform chunks of 64 (64 bytes)
    for (i = firstpart; i + 64 <= length; i += 64)
      transform(&message[i], state);
 
    index = 0;
  }
  else
    i = 0;
  // buffer remaining input
  charMemcpy(&buffer[index], &message[i], (length-i));

  // finalized --------------------------------------------------------------
  unsigned char bits[8];
  //encode(bits, count, 8);
  for (uint4_md5 i = 0, j = 0; j < 8; i++, j += 4) {
    bits[j] = count[i] & 0xff;
    bits[j+1] = (count[i] >> 8) & 0xff;
    bits[j+2] = (count[i] >> 16) & 0xff;
    bits[j+3] = (count[i] >> 24) & 0xff;
  }

  index = count[0] / 8 % 64;
  uint4_md5 padLen = (index < 56) ? (56 - index) : (120 - index);

  //sekwencyjny kod odniesienia to dwie ponizsze linijki
  //update(padding, padLen);
  //update(bits, 8);

  // i teraz zakladam �e przepisuje tylko pewien kawalek metody update inny dla padding i inny bits, bo w tym momencie to bedzie transformacja ostatniego bloku
  //dla padding - nie wykonana zostanie tranformacja (dopiero dla bits)
  //length = padLen, input = padding
  
  // compute number of bytes mod 64
  index = count[0] / 8 % 64; 
  // Update number of bits
  if ((count[0] += (padLen << 3)) < (padLen << 3))
    count[1]++;
  count[1] += (padLen >> 29);

  // buffer remaining input
  charMemcpy(&buffer[index], &padding[0], padLen); //padding najprawdopodobniej bedzie gdzies w pamieci dzielonej/shared

  //dla bits
  //length = 8, input = bits
  index = count[0] / 8 % 64; 
 // if ((count[0] += (8 << 3)) < (8 << 3))
 //   count[1]++;
//  count[1] += (length >> 29);

  firstpart = 64 - index;

  // fill buffer first, transform
  charMemcpy(&buffer[index], bits, firstpart);

  transform(buffer, state);
 
  // transform chunks of 64 (64 bytes)
//  for (i = firstpart; i + 64 <= length; i += 64)
 //   transform(&bits[i], state);

  // i na razie nie zeruje buffer funkcja memset bo nie wiem po co to/ czy to jest konieczne

  // encode(digest, state, 16);
  for (uint4_md5 i = 0, j = 0; j < 16; i++, j += 4) {
    digest[j] = state[i] & 0xff;
    digest[j+1] = (state[i] >> 8) & 0xff;
    digest[j+2] = (state[i] >> 16) & 0xff;
    digest[j+3] = (state[i] >> 24) & 0xff;
  }
  // i w tym momencie w digest powinien byc wygenerowany skrot
  charMemcpy(foundCollision, digest, 16);
}