#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

int main(int argc, char **argv){
	int a = 8;
	int b = 3;
	int c = 8;
	printf("a/b= %d\n", a/b);
	printf("a/c= %d\n", a/c);
	return 0;
}