#include <iostream>
#include <cuda_runtime.h>
#define xd 0.001
#define STEPS 10

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

float* concatenar(float *f1, float *f2, float *f3, float *f4, int N, int M){
		float *resultado = new float[N*M];
		int size = N * M /4;
		for (int i = 0; i < size; i ++){
				resultado[i + 0*size] = f1[i];
				resultado[i + 1*size] = f1[i];
				resultado[i + 2*size] = f1[i];
				resultado[i + 3*size] = f1[i]; 
		}
		return resultado;
}

/* Procesamiento CPU */
void CPU_1_step(float *f_in, float *f_out, int N, int M){
		int x,y;
		for (int i = 0; i < N*M; i++){
				x = i % N;
				y = i / N;
				f_out[i] = (f_in[modulo(x+1, N) + y*N] + f_in[modulo(x-1, N) + y*N]) /(2); // xd
				//f_out[i] = 2;
		}
}

void CPU(){
	int M, N;
	float *f_in, *f_out, *temp;
	clock_t t1, t2;
	double ms;

	Read(&f_in, &M, &N,0);
	//imprimir_malla(f_in, N, M);
	f_out = new float[N*M];

	t1 = clock();
	for(int step = 0; step< STEPS; step++){
			CPU_1_step(f_in, f_out, N,M);
			temp = f_out;
			f_out = f_in;
			f_in = temp;
	}
	f_out =f_in;
	t2 = clock();	
	ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	printf("Tiempo CPU: %f[ms]\n", ms);

	//imprimir_malla(f_out, N,M);
 	delete[] f_in;
 	delete[] f_out;
}



/*  Procesamiento GPU, 1 stream */
__global__ void kernel_1(float *f, float *f_out, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M){ //1 thread para cada fila
	    float anterior = f[modulo(-1, N) + tid*N];
			float actual = f[0 + tid*N];
			float siguiente; 
			for (int i = 0; i< N; i++){
					siguiente = f[modulo(i+1, N) + tid*N];
					f_out[i + tid*N] = (anterior + siguiente) / (2); //xd

					anterior = actual;
					actual = siguiente;
			}
	}
}
void GPU_1_stream(){	
		printf("gpu 1\n");
		cudaEvent_t ct1, ct2;	
		float dt;
		int M, N;
		float *f_host, *f_hostout, *f, *f_out, *temp;
		int gs, bs = 256;

		Read(&f_host, &M, &N,0);
		gs = (int)ceil((float) M / bs);    
		//imprimir_malla(f_host, N,M);

		cudaMalloc((void**)&f, M * N * sizeof(float));
		cudaMemcpy(f, f_host, M * N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&f_out, M * N * sizeof(float));
		//cudaMalloc((void**)&temp, M * N * sizeof(float));

		cudaEventCreate(&ct1);
		cudaEventCreate(&ct2);
		cudaEventRecord(ct1);


		// llamadas al kernel
		for (int i = 0 ; i< STEPS; i++){
			kernel_1<<<gs, bs>>>(f, f_out, N, M);
			temp = f_out;
			f_out = f;
			f = temp;
		}
		f_out =f;
	
	
		cudaEventRecord(ct2);
		cudaEventSynchronize(ct2);
		cudaEventElapsedTime(&dt, ct1, ct2);
		f_hostout = new float[M * N];
		cudaMemcpy(f_hostout, f, M * N * sizeof(float), cudaMemcpyDeviceToHost);

		//Write(f_hostout, M, N, "initial_S.txt\0");
		//imprimir_malla(f_hostout, N,M);
		std::cout << "Tiempo " << ": " << dt << "[ms]" << std::endl;
		cudaFree(f);
		//cudaFree(temp);
		cudaFree(f_out);
		delete[] f_host;
		delete[] f_hostout;
}



__global__ void kernel_2(float *f, float *f_out, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M/4){ //1 thread para cada fila
	    float anterior = f[modulo(-1, N) + tid*N];
			float actual = f[0 + tid*N];
			float siguiente; 
			for (int i = 0; i< N; i++){
					siguiente = f[modulo(i+1, N) + tid*N];
					f_out[i + tid*N] = (anterior + siguiente) / 2; //xd
					anterior = actual;
					actual = siguiente;
			}
	}
}


/*  Procesamiento GPU, 4 stream horizontal*/
void GPU_4_stream_horizontal(){
		printf("stream horizontal \n");
		cudaEvent_t ct1, ct2;	
		float dt;
		int M, N;
		float *f_host;
		float * f_in;
		float *f_out;
		int gs, bs = 256;

		//crear streams
		cudaStream_t str1, str2, str3, str4;
		cudaStreamCreate(&str1);
		cudaStreamCreate(&str2);
		cudaStreamCreate(&str3);
		cudaStreamCreate(&str4);

		Read(&f_host, &M, &N,1);
		gs = (int)ceil((float) (M/4) / bs);    
		//imprimir_malla(f_host, N,M);
		int size =  M/4 * N ;

		cudaMalloc(&f_in,  M * N* sizeof(float));
		cudaMalloc(&f_out, M * N* sizeof(float));
		float *out = new float[N*M];
		float *temp;

		//host to device
		cudaMemcpyAsync(&f_in[size*0], &f_host[size*0], size * sizeof(float), cudaMemcpyHostToDevice, str1);
		cudaMemcpyAsync(&f_in[size*1], &f_host[size*1], size * sizeof(float), cudaMemcpyHostToDevice, str2);
		cudaMemcpyAsync(&f_in[size*2], &f_host[size*2], size * sizeof(float), cudaMemcpyHostToDevice, str3);
		cudaMemcpyAsync(&f_in[size*3], &f_host[size*3], size * sizeof(float), cudaMemcpyHostToDevice, str4);
		

		//kernel calls
		cudaEventCreate(&ct1);
		cudaEventCreate(&ct2);
		cudaEventRecord(ct1);
	
		// llamadas al kernel
		for (int i = 0 ; i< STEPS; i++){
			kernel_2<<<gs, bs,0,str1>>>(f_in, f_out, N, M);
			kernel_2<<<gs, bs,0,str2>>>(&f_in[size*1], &f_out[size*1], N, M);
			kernel_2<<<gs, bs,0,str3>>>(&f_in[size*2], &f_out[size*2], N, M);
			kernel_2<<<gs, bs,0,str4>>>(&f_in[size*3], &f_out[size*3], N, M);
			temp = f_out;
			f_out = f_in;
			f_in = temp;
		}
		//f_out =f_in;
		
	
	
		cudaEventRecord(ct2);
		cudaEventSynchronize(ct2);
		cudaEventElapsedTime(&dt, ct1, ct2);


		//device to host
		cudaMemcpyAsync(&out[size*0], &f_in[size*0], size * sizeof(float), cudaMemcpyDeviceToHost,str1);
		cudaMemcpyAsync(&out[size*1], &f_in[size*1], size * sizeof(float), cudaMemcpyDeviceToHost,str2);
		cudaMemcpyAsync(&out[size*2], &f_in[size*2], size * sizeof(float), cudaMemcpyDeviceToHost,str3);
		cudaMemcpyAsync(&out[size*3], &f_in[size*3], size * sizeof(float), cudaMemcpyDeviceToHost,str4);

		//Write(out, M, N, "initial_S.txt\0");
		cudaDeviceSynchronize();
		//imprimir_malla(out, N,M);
		std::cout << "Tiempo " << ": " << dt << "[ms]" << std::endl;
		cudaFree(f_host);
		cudaFree(f_in);
		cudaFree(f_out);
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
		
	//ejecucion cpu
	//CPU(); //212

	GPU_1_stream(); //23 1784

	//GPU_4_stream_horizontal(); //23 1442

	//GPU_4_stream_vertical();
	
	return 0;
}