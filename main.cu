
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "md5.cuh"
#include <iostream>

int main()
{
	unsigned char* result = (unsigned char*) malloc(16*sizeof(char));
	runMD5(result);
	printResult(result);

	//to na dole, po to ¿eby konsola siê nie wy³¹cza³a od razu
	int x;
	std::cin >> x;
    return 0;
}
