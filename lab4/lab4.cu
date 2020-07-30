#include <iostream>
#include <cuda_runtime.h>
#define xd 0.001

/*  Lectura Archivo */
void Read(float **f, int *M, int *N, int tipo=0) {
    FILE *fp;
    fp = fopen("initial.txt\0", "r");
    fscanf(fp, "%d %d\n", M, N);
		float *f1;

    int size = (*M) * (*N);
		if (tipo == 0)
    	f1 = new float[size];
		else
			cudaMallocHost(&f1, sizeof(float)* size); //pinned memory
		
		
		for(int i = 0; i < size; i++){
					fscanf(fp, "%f ", &(f1[i]));
					// printf("%d ", i*4 + x);
		}
	
    fclose(fp);
    *f = f1;
}



/*  Escritura de archivo initial con array */
void Write(int *f, int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", N, M);
    int Largo = M*N;

		for(int i = 0; i < Largo-1; i++){
				fprintf(fp, "%d ", f[i]);
			
		}
		fprintf(fp, "%d\n", f[Largo-1]);
    //printf("\n");
    fclose(fp);
}


//funcion auxiliar %, funciona con entradas negativas
__host__ __device__ int modulo(int a, int b){
    //a%b
    if (a >= 0){
        return a %b;
    }
    return b + a;
}

void imprimir_malla(float *f, int N , int M){
		for(int j = 0; j< M; j ++){
			for(int i = 0; i< N; i ++){
						printf("%.2f ", f[i+ j*M]);
				}
				printf("\n");
		}
		printf("-----\n");
}

/* Procesamiento CPU */
void CPU_1_step(float *f_in, float *f_out, int N, int M){
		int x,y;
		for (int i = 0; i < N*M; i++){
				x = i % N;
				y = i / N;
				f_out[i] = (f_in[modulo(x+1, N) + y*N] + f_in[modulo(x-1, N) + y*N]) /(2*xd); // xd
				//f_out[i] = 2;
		}
		
}

void CPU(float *f_in, float *f_out, int N, int M){
	clock_t t1, t2;
	float *temp;
	double ms;
	Read(&f_in, &M, &N,0);
	imprimir_malla(f_in, N, M);
	f_out = new float[N*M];

	t1 = clock();
	for(int step = 0; step< 10; step++){
			CPU_1_step(f_in, f_out, N,M);
			temp = f_out;
			f_out = f_in;
			f_in = temp;
	}
	f_out =f_in;
	t2 = clock();	
	ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	printf("Tiempo CPU: %f[ms]\n", ms);

	imprimir_malla(f_out, N,M);
 	delete[] f_in;
 	delete[] f_out;
}



/*  Procesamiento GPU, 1 stream */
__global__ void kernel1(int *f, int *f_out, int X, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M*N){
	}
}
void GPU_1_stream(){
	//	gs = (int)ceil((float) M * N / bs);    
	//		cudaMalloc((void**)&f, M * N * sizeof(int));
	//		cudaMemcpy(f, f_host, M * N * sizeof(int), cudaMemcpyHostToDevice);
	//		cudaMalloc((void**)&f_out, M * N * sizeof(int));
	//		cudaMalloc((void**)&temp, M * N * sizeof(int));
//
	//		cudaEventCreate(&ct1);
	//		cudaEventCreate(&ct2);
	//		cudaEventRecord(ct1);
//
	//		// llamadas al kernel
	//	
	//	
	//		cudaEventRecord(ct2);
	//		cudaEventSynchronize(ct2);
	//		cudaEventElapsedTime(&dt, ct1, ct2);
	//		f_hostout = new int[M * N * X];
	//		cudaMemcpy(f_hostout, f, M * N * X * sizeof(int), cudaMemcpyDeviceToHost);
//
//
	//		Write(f_hostout, M, N, "initial_S.txt\0");
	//		metodo = "SoA";
//
	//		std::cout << "Tiempo " << metodo << ": " << dt << "[ms]" << std::endl;
	//		cudaFree(f);
	//		cudaFree(temp);
	//		cudaFree(f_out);
	//		delete[] f_host;
	//		delete[] f_hostout;
	//	}
	printf("gpu 1");
		
}



/*  Procesamiento GPU, 4 stream horizontal*/
void GPU_4_stream_horizontal(){
			printf("gpu 1");
}
__global__ void kernel2(int *f, int *f_out, int X, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M*N){
	}
}



/*  Procesamiento GPU, 4 stream vertical*/
void GPU_4_stream_vertical(){
			printf("gpu 1");
}
__global__ void kernel3(int *f, int *f_out, int X, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M*N){
	}
}


//--------------------------------------------------------------------------------


/*  Codigo Principal */
int main(int argc, char **argv){
		
	//inicializacion de variables
	cudaEvent_t ct1, ct2;

	float dt;
	const char *metodo;
	int M, N;
  float *f_host, *f_hostout, *f, *f_out, *temp;
	int gs, bs = 256;

	
	
	//ejecucion cpu
	CPU(f_host,f_out, N,M);

	//GPU_1_stream();

	//GPU_4_stream_horizontal();

	//GPU_4_stream_vertical();
	
	return 0;
}