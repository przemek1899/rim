
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "helper_cuda.h"
#include "md5.cuh"
//#include "md5cpu.h"
#include <iostream>
#include <stdio.h>
#include "string.h"
#include <fstream>
using std::cout; using std::endl;
void printResult(unsigned char* result){
	char buf[33];
	for (int i=0; i<16; i++)
		sprintf(buf+i*2, "%02x", result[i]);
	buf[32]=0;
	printf("%s\n", buf);
}

void charMemcpyCPU(unsigned char *buffer, unsigned char *data, int length){

	int i;
	for(i=0; i<length; i++){
		buffer[i] = data[i];
	}
}


const std::string* cpuHash;
int main()
{
	//wczytanie bloku 128 bajtów danych z pliku
	std::ifstream plik1("mm", std::ios::binary);
		
		
		char cstr1[128];
		
		
		
		unsigned char message[128];

		
		plik1.read(cstr1, 128);
		printf("Dane wejsciowe  M oraz N  wczytane z pliku :\n");


		for(int i=0; i<128; i=i+4) { 
		printf("< %-+d   %-+d   %-+d   %-+d    >  \n", 
				(int)cstr1[i],(int)cstr1[i+1],  
				(int)cstr1[i+2],(int)cstr1[i+3]);
	    }

		printf("Przepisanie wczytanej wiadomosci do wektora wiadomosci W \n");
		for(int i=0; i<128; i++){
			message[i]=(unsigned char)cstr1[i];
		}

		printf(" \n Roznice miêdzy dwoma wektroami \n");
		for(int i=0; i<128; i++) { 
		printf("< nr indeksu %d     roznica  %-+d      >  \n", 
			i,	((char)cstr1[i]-(char)message[i]));
	    }
		
		//wykorzystywany do liczenia md5 na CPU
		std::string str1=std::string(cstr1, 128);

	

	unsigned char* result = (unsigned char*) malloc(16*sizeof(unsigned char));
	unsigned char* orygHash = (unsigned char*) malloc(16*sizeof(unsigned char));
	
	

	//w³aœciwa funkcja 
	runMD5(message,orygHash, result, 128);
	
	
	printf("Dane wyjsciowe \n");
		for(int i=0; i<128; i=i+4) { 
		printf("< %-+d   %-+d   %-+d   %-+d    >  \n", 
				(int)(char)message[i],(int)(char)message[i+1],  
				(int)(char)message[i+2],(int)(char)message[i+3]);
	    }
		printf(" \n Roznice miedzy dwoma wektroami \n");
		for(int i=0; i<128; i++) { 
		printf("< nr indeksu %d     roznica  %-+d      >  \n", 
			i,	((char)cstr1[i]-(char)message[i]));
	    }

	printf(" \n Skrot wiadomoœci oryginalnej obliczony przez biblioteke CPU: \n");
//	cout<<md5(str1)<<"     "<<endl;
	printf(" \n Skrot wiadomoœci oryginalnej obliczony przez biblioteke GPU: \n");
	printResult(orygHash);
	printf(" \n Skrot wiadomoœci kolizyjnej obliczony przez biblioteke GPU: \n");
	printResult(result);
	
	//to na dole, po to ¿eby konsola siê nie wy³¹cza³a od razu

	
	
    return 0;
}
