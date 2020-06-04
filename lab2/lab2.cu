#include <iostream>
#include <cuda_runtime.h>

/*  Lectura Archivo */
void Read(int **f, int *M, int *N, const char *filename, int X, int tipo) {
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", N, M);

    int imsize = (*M) * (*N) * X;
    int* f1 = new int[imsize];
    int Lres = (*M) * (*N) / X;
    int Largo = (*M) * (*N);

    if (tipo == 0){ // AoS
		for(int x=0; x<X; x++){
			for(int i = 0; i < Largo; i++){
	        	fscanf(fp, "%d ", &(f1[i*4 + x]));
		        printf("%d ", f1[i*4 + x]);
		        // printf("%d ", i*4 + x);
			}
			printf("\n");
	    }
	    printf("\n");
	    // Datos M = 6, N = 4

	    //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 											23
	    // 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 											47
	    // 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 											71
	    // 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 											95
	    
	    //  0 24 48 72  1 25 49 73  2 26 50 74  3 27 51 75  4 28 52 76   5 29 53 77  6 30 54 78  7 31 55 79  8 32 56 80 ... 95

	    //  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1 ..    		 3	(x) < X
	    //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23  0  1  2  3  4  5  6  7  8 ...			23 (i) < N*M

	    //  i*4 + x
	    //  0  4  8 12 16 20 24 28 		....  		1  5  9 13 17 21 25 29    ......	2  6 10 14 18 22 	....	3  7 11 15 19 .. 

	} else{ // SoA 
		for(int j=0; j<X; j++){
	    	for(int i = 0; i < Largo-1; i++){
		        fscanf(fp, "%d ", &(f1[i + j*Largo]));
	    	}
		    fscanf(fp, "%d\n", &(f1[Largo-1 + j*Largo]));
	    }
	}
    fclose(fp);
    *f = f1;
}

/*  Escritura de archivo initial con array */
void Write(int *f, int Lres, int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", N, M);
    int Largo = M*N;
    for(int j=0; j<4; j++){
    	for(int i = 0; i < Largo-1; i++){
	        fprintf(fp, "%d ", f[i + j*Largo]);
	    	printf("%d ", f[i + j*Largo]);
	    }
	    fprintf(fp, "%d\n", f[Largo-1 + j*Largo]);
	    printf("%d\n", f[Largo-1 + j*Largo]);
    }
    fclose(fp);
}

/*  Procesamiento GPU AoS Coalisiones */
__global__ void kernelAoS_col(int *f, int *f_out, int Lres, int X, int Largo){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
}

/*  Procesamiento GPU AoS Streaming */
__global__ void kernelAoS_stream(int *f, int *f_out, int Lres, int X, int Largo){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
}

/*  Procesamiento GPU SoA Coalisiones */
__global__ void kernelSoA_col(int *f, int *f_out, int Lres, int X, int Largo){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
}

/*  Procesamiento GPU SoA Streaming */
__global__ void kernelSoA_stream(int *f, int *f_out, int Lres, int X, int Largo){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
}

/*  Codigo Principal */
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	cudaEvent_t ct1, ct2;
	float dt;
	// N eje y, M eje x
	int M, N;
    int *f_host, *f_hostout, *f, *f_out;
    char filename[15] = "initial.txt\0";
	int gs, bs = 256;
	int X = 4;
	int Lres;

	// Iteramos los 2 metodos SoA y AoS
    for (int i=0; i<1; i++){
    	Read(&f_host, &M, &N, filename, X, i);

    	Lres = M*N/4;

	    gs = (int)ceil((float) M * N * X / bs);    
	    cudaMalloc((void**)&f, M * N * X * sizeof(int));
	    cudaMemcpy(f, f_host, M * N * X * sizeof(int), cudaMemcpyHostToDevice);
	    cudaMalloc((void**)&f_out, M * N * X * sizeof(int));
	    
    	Write(f_host, Lres, M, N, "initial_f.txt\0");

	    cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);

	    // Iteraciones de time step
	    for (int j=0; j<0; j++){
	    	if (i==0){
	    		kernelAoS_col<<<gs, bs>>>(f, f_out, Lres, X, N*M);
	    		kernelAoS_stream<<<gs, bs>>>(f, f_out, Lres, X, N*M);
	    	}
	    	else{
	    		kernelSoA_col<<<gs, bs>>>(f, f_out, Lres, X, N*M);
	    		kernelSoA_stream<<<gs, bs>>>(f, f_out, Lres, X, N*M);
	    	}
	    }

	    cudaEventRecord(ct2);
	    cudaEventSynchronize(ct2);
	    cudaEventElapsedTime(&dt, ct1, ct2);
	    std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;

	    f_hostout = new int[M * N * X];
	    cudaMemcpy(f_hostout, f_out, M * N * X * sizeof(int), cudaMemcpyDeviceToHost);

	    // Write(f_hostout, Lres, M, N, "initial_f.txt\0");

    	cudaFree(f);
    	cudaFree(f_out);
    	delete[] f_host;
    	delete[] f_hostout;
	}
	return 0;
}