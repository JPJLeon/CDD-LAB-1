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

    if (tipo == 0){
		for(int j=0; j<4; j++){
	    	for(int i = 0; i < Largo-1; i++){
		        fscanf(fp, "%d ", &(f1[i + j*Largo]));
		        printf("%d ", f1[i + j*Largo]);
	    	}
		    fscanf(fp, "%d\n", &(f1[Largo-1 + j*Largo]));
		    printf("%d\n", f1[Largo-1 + j*Largo]);
	    }
	    // Datos M = 6, N = 4, X = 4
	    // Lres = N*M/X = 6
	    
	} else{
		for(int j=0; j<4; j++){
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


	    /*
	     *  Parte GPU
	     */
	    gs = (int)ceil((float) M * N * X / bs);
	        
	    cudaMalloc((void**)&f, M * N * X * sizeof(int));
	    cudaMemcpy(f, f_host, M * N * X * sizeof(int), cudaMemcpyHostToDevice);
	    cudaMalloc((void**)&f_out, M * N * X * sizeof(int));
	    
    	// Write(f_host, Lres, M, N, "initial_f.txt\0");

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

	    Write(f_hostout, Lres, M, N, "initial_f.txt\0");

    	cudaFree(f);
    	cudaFree(f_out);
    	delete[] f_host;
    	delete[] f_hostout;
	}
	return 0;
}