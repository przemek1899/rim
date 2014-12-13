
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "md5.cuh"
#include <iostream>
#include <fstream>

int main()
{

	// te jego niby wczytanie pliku
	//wczytanie bloku 128 bajtów danych z pliku
	std::ifstream plik1("mm", std::ios::binary);
	//char* message = (char*) malloc(128*sizeof(char));
	char *cstr1 = new char[128];
	plik1.read(cstr1, 128);
	//wyœwietlenie danych jako int.
	uint1_md5 message[11] = {'a','l','a',' ', 'm', 'a',' ','k','o','t','a'};

	unsigned char* result = (unsigned char*) malloc(16*sizeof(char));
	runMD5(result, (unsigned char *) cstr1, 128);
	printResult(result);

	//to na dole, po to ¿eby konsola siê nie wy³¹cza³a od razu
	int x;
	//std::cin >> x;
    return 0;
}
