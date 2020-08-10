#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>

// Variables globales GPU y CPU
#define l_kernel 2
#define stride 2

/******************************
 *  Procesamiento Matriz CPU  *
 ******************************/

/*
 *  Funcion Max
 */
float MaxCPU(float A, float B){
	float result = A > B ? A : B;
	return result;
}

/*
 *  Lectura Archivo
*/
void Read(float** R, float** G, float** B, int *M, int *N, const char *filename, int tipo) {
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", M, N);

    int imsize = (*M) * (*N);
	
	float* R1, * G1, * B1;

    if (tipo == 0){ // Lectura normal
    	R1 = new float[imsize];
		G1 = new float[imsize];
		B1 = new float[imsize];
		for(int i = 0; i < imsize; i++)
		    fscanf(fp, "%f ", &(R1[i]));
		for(int i = 0; i < imsize; i++)
		    fscanf(fp, "%f ", &(G1[i]));
		for(int i = 0; i < imsize; i++)
		    fscanf(fp, "%f ", &(B1[i]));
	}
	else if (tipo == 3){ //lectura en pinned memory
		cudaMallocHost(&R1, sizeof(float)* imsize);
		cudaMallocHost(&G1, sizeof(float)* imsize);
		cudaMallocHost(&B1, sizeof(float)* imsize);
		for(int i = 0; i < imsize; i++)
		    fscanf(fp, "%f ", &(R1[i]));
		for(int i = 0; i < imsize; i++)
		    fscanf(fp, "%f ", &(G1[i]));
		for(int i = 0; i < imsize; i++)
		    fscanf(fp, "%f ", &(B1[i]));
	}
    fclose(fp);
    *R = R1; *G = G1; *B = B1;
}

/*
 *  Escritura Archivo
 */
void Write(float* out, int M_out, int N_out, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M_out, N_out);
    for(int i = 0; i < M_out*N_out-1; i++)
        fprintf(fp, "%f ", out[i]);
    fprintf(fp, "%f\n", out[M_out*N_out-1]);
    for(int i = 0; i < M_out*N_out-1; i++)
        fprintf(fp, "%f ", out[i]);
    fprintf(fp, "%f\n", out[M_out*N_out-1]);
    for(int i = 0; i < M_out*N_out-1; i++)
        fprintf(fp, "%f ", out[i]);
    fprintf(fp, "%f\n", out[M_out*N_out-1]);
    fclose(fp);
}

/*
 *  Imprimir Array como matriz
 */
void ShowMatrix(float *matrix, int N, int M) {
    for(int i = 0; i < N; i++){
    	for(int j = 0; j < M; j++)
    		printf("%.4f ", matrix[j + i*M]);
    	printf("\n");
    }
    printf("\n");
}

/*
 *  Suma de Matrices R,G,B y Funcion de activacion RELU
 */
void SumaMatrizCPU(float **out, float *R, float *G, float *B, int M, int N){
	float* sum = new float[M*N];
	for(int i=0; i < M*N; i++){
		sum[i] = (R[i]+G[i]+B[i])/3.0;
	}
	*out = sum;
}

/*
 *  Funcion de activacion RELU
 */
void ReluCPU(float *out, int M, int N){
	for(int i=0; i < M*N; i++){
		if(out[i] < 0){
			out[i] = 0;
		} else if(out[i] > 1){
			out[i] = 1;
		}
	}
}

/*
 *  Funcion Max pooling 2x2
 */
void PoolingCPU(float **out, int *M, int *N){
	int new_N, new_M;
	float v1, v2, v3, v4;
	new_N = (*N)/2;
	new_M = (*M)/2;
	// printf("new_M: %d new_N: %d\n", new_M, new_N);
	float* temp = new float[new_N*new_M];
	for(int i=0; i < new_M; i++){
		for(int j=0; j < new_N; j++){
			// printf("v1: %d\n", j*2 + i*2*(*N));
			v1 = (*out)[j*2 + i*2*(*N)];
			v2 = (*out)[j*2 + 1 + i*2*(*N)];
			v3 = (*out)[j*2 + (i+1)*(*N)];
			v4 = (*out)[j*2 + 1 + (i+1)*(*N)];
			temp[j + i*new_N] = MaxCPU(MaxCPU(v1, v2), MaxCPU(v3, v4));
		}
	}
	*out = temp;
	*N = new_N; *M = new_M;
}

/*
 *  "Producto" Matricial sub_A * kernel = C
 *  id: id del primer elemento de la submatriz, N: ancho matriz R
 */
float Product_Matrix(float *A, float *B, int N_original, int id){
	int col, row, idx_kernel;
	float count;
	col = id%N_original;
	row = id/N_original;
	count = 0.0;
	// Recorremos stride
	idx_kernel = 0;
	for(int i=row; i < row + l_kernel; i++){
		for(int j=col; j< col + l_kernel; j++){
			int id_casilla = j + i*N_original;
			// printf("%.1f x %.1f\n", A[id_casilla], B[idx_kernel]);
			count += A[id_casilla] * B[idx_kernel];
			idx_kernel += 1;
		}
	}
	return count;
}

/*
 *  Convolucion de A y kernel (recorre la primera matriz y hace el producto matricial por cada elemento)
 */
void ConvolucionCPU(float *A, float **out, float *kernel, int Mres, int Nres, int N_original){
	float* temp = new float[Nres*Mres];
	int count_output = 0;
	int i = 0;
	while(i < N_original*(N_original-1)){
		if((i/N_original)%2 == 0){
			temp[count_output] = Product_Matrix(A, kernel, N_original, i);
			// printf("i: %d out:%d\n", i, count_output);
			// printf("fila: %d\n", (i/N_original));
			count_output++;
			i = i+stride;
		} else{
			i = i+N_original;
		}
	}
	*out = temp;
}

void cnn_CPU(){
	int M, N;
	clock_t t1, t2;
	double ms;
	float array[l_kernel*l_kernel] = {1, 0, 1, -2}; // Conjunto de kernel(matrices) a usar
	// float array[l_kernel*l_kernel] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Conjunto de kernel(matrices) a usar
	// float array[l_kernel*l_kernel] = {-5, 5, 0, -5, 5, 0,-5, 5, 0}; // Conjunto de kernel(matrices) a usar
	// float array[l_kernel*l_kernel] = {-5, -5, -5, 5, 5, 5, 0, 0, 0}; // Conjunto de kernel(matrices) a usar
	float *kernel = new float[l_kernel*l_kernel];
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout, *Ghostout, *Bhostout;
    // Lectura de archivo
	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img.txt", 0);
	kernel = &array[0];
	printf("Kernel:\n");
	ShowMatrix(kernel, l_kernel, l_kernel);

	float *output_image; // Conjunto de imagenes(matrices) de salida por kernel
	printf("Matriz original: %d x %d\n", M, N);
	t1 = clock();
	// ShowMatrix(Rhost, M, N);
	// Por cada proceso de convolucion
    for(int c=0; c<1; c++){
    	// printf("\n########## Convolucion %d ###########\n", c+1);
		// Actualizamos N,M si aun se puede
		int N_original = N;
		printf("M: %d N: %d\n", M, N);
		N = N/stride;
		M = M/stride;
		// Si es el primero se suman las matrices RGB resultantes
		if(c == 0){
			// ShowMatrix(Rhost, M + l_kernel -1, N + l_kernel -1);
			ConvolucionCPU(Rhost, &Rhostout, kernel, M, N, N_original);
			ConvolucionCPU(Ghost, &Ghostout, kernel, M, N, N_original);
			ConvolucionCPU(Bhost, &Bhostout, kernel, M, N, N_original);
			// ShowMatrix(Rhostout, M, N);
			SumaMatrizCPU(&output_image, Rhostout, Ghostout, Bhostout, M, N);
		} else {
			// ShowMatrix(output_image, stride*M, stride*N);
			ConvolucionCPU(output_image, &output_image, kernel, M, N, N_original);
		}
		printf("Matriz Convolucion %d: %d x %d\n", c+1, M, N);
		// ShowMatrix(output_image, M, N);
    }

	ReluCPU(output_image, M, N);
    // PoolingCPU(&output_image, &M, &N);
    // printf("Matriz Pooling: %d x %d\n", M, N);

    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    std::cout << "Tiempo CPU: " << ms << "[ms]" << std::endl;
	// printf("Imagen pooling: %d x %d\n", M, N);
	// ShowMatrix(output_image, M, N);

    // printf("Imagen salida: %d x %d\n", M, N);
	// ShowMatrix(output_image, M, N);
	Write(output_image, M, N, "ResultadoCPU.txt");
	delete[] Rhost; delete[] Ghost; delete[] Bhost;
	delete[] Rhostout; delete[] Ghostout; delete[] Bhostout, delete[] output_image;
}


__global__ void kernel_sum(float *Rin, float *Gin, float *Bin, float *out, int Mres, int Nres){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < Mres*Nres){
		out[tid] = (Rin[tid] + Gin[tid] + Bin[tid])/3.0;
	}
}

__global__ void kernel_relu(float *out, int Mres, int Nres){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < Mres*Nres){
		if(out[tid] < 0){
			out[tid] = 0.0;
		} else if(out[tid] > 1){
			out[tid] = 1.0;
		}
	}
}

__global__ void kernel_poolingAoS(float *in, float *out, int Mres, int Nres, int N_original){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < Nres*Mres){ //1 thread para cada pixel de salida
		float max = 0.0;
		int x, y, col = 0;
		x = (tid%Nres);
		y = (tid/Nres);

		// Si N_original es impar, el pooling del borde se almacena con este thread
		if(!N_original%2 && x+1 == N_original-1){
			col = tid%Mres;
		}
		float valores[4] = {
			in[x*2 + y*2*(N_original)],
			in[x*2 + 1 + y*2*(N_original)],
			in[x*2 + (y+1)*(N_original)],
			in[x*2 + 1 + (y+1)*(N_original)]
		};
		for (int i = 0; i< 4; i++){
			if (valores[i] > max){
				max = valores[i];
			}
		}
		out[tid - col] = max;
	}
}

__global__ void kernel_convolucionAoS(float *in, float *out, float *kernel_dev, int Mres, int Nres){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < Mres*Nres){
		int x, y, N_original;
		x = 1 + tid%Nres; //coordenaas del centro de cada sub_matriz
		y = 1 + tid/Nres;
		N_original = Nres*2;
		float suma = 0;
		int indice_sub_matriz, indice_kernel;
		for (int i = -1; i<=1 ; i++){
			for (int j = -1; j <= 1; j++){
				indice_sub_matriz = (x+i)*stride + (y+j)*stride*(N_original);
				indice_kernel = (1+i) + (1+j)*3;
				suma += in[indice_sub_matriz] * kernel_dev[indice_kernel];
			}
		}
		// printf("%f\n", suma);
		out[tid] = suma;
	}
}

__global__ void kernel_convolucion_AoS_MC(float *in, float *out, int Mres, int Nres){

	__shared__ int kernel_local[4];
	kernel_local[0] = 1;
	kernel_local[1] = 0;
	kernel_local[2] = 1;
	kernel_local[3] = -2;

	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if (tid < Mres*Nres){
		int x, y, N_original;
		x = 1 + tid%Nres; //coordenaas del centro de cada sub_matriz
		y = 1 + tid/Nres;
		N_original = Nres*2;
		float suma = 0;
		int indice_sub_matriz, indice_kernel;
		for (int i = -1; i<=1 ; i++){
			for (int j = -1; j <= 1; j++){
				indice_sub_matriz = (x+i)*stride + (y+j)*stride*(N_original);
				indice_kernel = (1+i) + (1+j)*3;
				suma += in[indice_sub_matriz] * kernel_local[indice_kernel];
			}
		}
		// printf("%f\n", suma);
		out[tid] = suma;
	}
}

/*
 *  Procesamiento AoS GPU
 */

void AoS_GPU(){
	//procesamiento de las convoluciones con 3 streams, 1 por cada color
	cudaEvent_t ct1, ct2;	
	float dt;

	int M, N, Mres, Nres;
	int gs, bs = 256;
	float array[l_kernel*l_kernel] = {1, 0, 1, -2};
	// float array[l_kernel*l_kernel] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Conjunto de kernel(matrices) a usar
	float *kernel = new float[l_kernel*l_kernel];
	kernel = &array[0];

    float *Rhost, *Ghost, *Bhost, *hostout;
    float *Rdev_in, *Gdev_in, *Bdev_in, *Rdev_out, *Gdev_out, *Bdev_out, *kernel_dev;

	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img.txt", 0);
	if(stride == 1){
		Mres = M - l_kernel + 1;
		Nres = N - l_kernel + 1;
	} else{
		Mres = M/2;
		Nres = N/2;
	}
	gs = (int)ceil((float) Mres*Nres / bs);

	// kernel gpu
	cudaMalloc((void**)&kernel_dev, l_kernel * l_kernel * sizeof(float));

	// Arrays de entrada
	cudaMalloc((void**)&Rdev_in, M * N * sizeof(float));
    cudaMalloc((void**)&Gdev_in, M * N * sizeof(float));
    cudaMalloc((void**)&Bdev_in, M * N * sizeof(float));
    
    // Array de salida
    cudaMalloc((void**)&Rdev_out, Mres * Nres * sizeof(float));
    cudaMalloc((void**)&Gdev_out, Mres * Nres * sizeof(float));
    cudaMalloc((void**)&Bdev_out, Mres * Nres * sizeof(float));
    
    // Copiar en memoria global de GPU
	cudaMemcpy(Rdev_in, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Gdev_in, Ghost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Bdev_in, Bhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(kernel_dev, kernel, l_kernel * l_kernel * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);

	//kernel calls
	for(int c=0; c<1; c++){
		if(c == 0){
			//convolucion
			kernel_convolucionAoS<<<gs, bs>>>(Rdev_in, Rdev_out, kernel_dev, Mres, Nres);
			kernel_convolucionAoS<<<gs, bs>>>(Gdev_in, Gdev_out, kernel_dev, Mres, Nres);
			kernel_convolucionAoS<<<gs, bs>>>(Bdev_in, Bdev_out, kernel_dev, Mres, Nres);
			// Unir canales
			kernel_sum<<<gs, bs>>>(Rdev_out, Gdev_out, Bdev_out, Rdev_out, Mres, Nres);
		} else{
			if(stride == 1){
				Mres = M - l_kernel + 1;
				Nres = N - l_kernel + 1;
			} else{
				Mres = Mres/2;
				Nres = Nres/2;
			}
			gs = (int)ceil((float) Mres*Nres / bs);
			//convolucion
			kernel_convolucionAoS<<<gs, bs>>>(Rdev_out, Rdev_out, kernel_dev, Mres, Nres);
		}
	}
	kernel_relu<<<gs, bs>>>(Rdev_out, Mres, Nres);
	printf("Imagen salida: %d x %d\n", Mres, Nres);
	int N_original = Nres;
	Nres = Nres/2;
	Mres = Mres/2;
	gs = (int)ceil((float) Mres*Nres / bs);
	Rdev_in = Rdev_out;
	kernel_poolingAoS<<<gs, bs>>>(Rdev_in, Rdev_out, Mres, Nres, N_original);

	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);

	hostout = new float[Mres*Nres];
	cudaMemcpy(hostout, Rdev_out, Mres * Nres * sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("Imagen salida: %d x %d\n", Mres, Nres);
	// ShowMatrix(hostout, Mres, Nres);

	Write(hostout, Mres, Nres, "ResultadoAoS.txt\0");

	std::cout << "Tiempo AoS" << ": " << dt << "[ms]" << std::endl;
	cudaFree(Rdev_in); cudaFree(Gdev_in); cudaFree(Bdev_in);
	cudaFree(Rdev_out); cudaFree(Gdev_out); cudaFree(Bdev_out);
	delete[] Rhost; delete[] Ghost; delete[] Bhost;
	delete[] hostout;
}

void AoS_MC_GPU(){
	//procesamiento de las convoluciones con 3 streams, 1 por cada color
	cudaEvent_t ct1, ct2;	
	float dt;

	int M, N, Mres, Nres;
	int gs, bs = 256;
	//float array[l_kernel*l_kernel] = {1, 0, 1, -2};
	// float array[l_kernel*l_kernel] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Conjunto de kernel(matrices) a usar
	//float *kernel = new float[l_kernel*l_kernel];
	//kernel = &array[0];

    float *Rhost, *Ghost, *Bhost, *hostout;
    float *Rdev_in, *Gdev_in, *Bdev_in, *Rdev_out, *Gdev_out, *Bdev_out; //*kernel_dev;

	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img.txt", 0);
	if(stride == 1){
		Mres = M - l_kernel + 1;
		Nres = N - l_kernel + 1;
	} else{
		Mres = M/2;
		Nres = N/2;
	}
	gs = (int)ceil((float) Mres*Nres / bs);

	// kernel gpu
	//cudaMalloc((void**)&kernel_dev, l_kernel * l_kernel * sizeof(float));

	// Arrays de entrada
	cudaMalloc((void**)&Rdev_in, M * N * sizeof(float));
    cudaMalloc((void**)&Gdev_in, M * N * sizeof(float));
    cudaMalloc((void**)&Bdev_in, M * N * sizeof(float));
    
    // Array de salida
    cudaMalloc((void**)&Rdev_out, Mres * Nres * sizeof(float));
    cudaMalloc((void**)&Gdev_out, Mres * Nres * sizeof(float));
    cudaMalloc((void**)&Bdev_out, Mres * Nres * sizeof(float));
    
    // Copiar en memoria global de GPU
	cudaMemcpy(Rdev_in, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Gdev_in, Ghost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(Bdev_in, Bhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	//cudaMemcpy(kernel_dev, kernel, l_kernel * l_kernel * sizeof(float), cudaMemcpyHostToDevice);

	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);

	//kernel calls
	for(int c=0; c<1; c++){
		if(c == 0){
			//convolucion
			kernel_convolucion_AoS_MC<<<gs, bs>>>(Rdev_in, Rdev_out, Mres, Nres);
			kernel_convolucion_AoS_MC<<<gs, bs>>>(Gdev_in, Gdev_out, Mres, Nres);
			kernel_convolucion_AoS_MC<<<gs, bs>>>(Bdev_in, Bdev_out, Mres, Nres);
			// Unir canales
			kernel_sum<<<gs, bs>>>(Rdev_out, Gdev_out, Bdev_out, Rdev_out, Mres, Nres);
		} else{
			if(stride == 1){
				Mres = M - l_kernel + 1;
				Nres = N - l_kernel + 1;
			} else{
				Mres = Mres/2;
				Nres = Nres/2;
			}
			gs = (int)ceil((float) Mres*Nres / bs);
			//convolucion
			kernel_convolucion_AoS_MC<<<gs, bs>>>(Rdev_out, Rdev_out, Mres, Nres);
		}
	}
	kernel_relu<<<gs, bs>>>(Rdev_out, Mres, Nres);
	printf("Imagen salida: %d x %d\n", Mres, Nres);
	int N_original = Nres;
	Nres = Nres/2;
	Mres = Mres/2;
	gs = (int)ceil((float) Mres*Nres / bs);
	Rdev_in = Rdev_out;
	kernel_poolingAoS<<<gs, bs>>>(Rdev_in, Rdev_out, Mres, Nres, N_original);

	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);

	hostout = new float[Mres*Nres];
	cudaMemcpy(hostout, Rdev_out, Mres * Nres * sizeof(float), cudaMemcpyDeviceToHost);
	
	printf("Imagen salida: %d x %d\n", Mres, Nres);
	// ShowMatrix(hostout, Mres, Nres);

	Write(hostout, Mres, Nres, "Resultado_AoS_MC.txt\0");

	std::cout << "Tiempo AoS con MC" << ": " << dt << "[ms]" << std::endl;
	cudaFree(Rdev_in); cudaFree(Gdev_in); cudaFree(Bdev_in);
	cudaFree(Rdev_out); cudaFree(Gdev_out); cudaFree(Bdev_out);
	delete[] Rhost; delete[] Ghost; delete[] Bhost;
	delete[] hostout;
}

/*
 *  Procesamiento Streams GPU
 */

/*
 *  Codigo Principal
 */

// void streams (){
// 	//procesamiento de las convoluciones con 3 streams, 1 por cada color
// 	cudaEvent_t ct1, ct2;	
// 	float dt;

// 	int M, N;
// 	int gs, bs = 256;

// 	float kernel[l_kernel*l_kernel] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Conjunto de kernel(matrices) a usar
//     float *Rhost, *Ghost, *Bhost;
//     float *Rhostout;

//     Rhostout = new float[l_kernel*l_kernel];

// 	//crear streams
// 	cudaStream_t str1, str2, str3;
// 	cudaStreamCreate(&str1);
// 	cudaStreamCreate(&str2);
// 	cudaStreamCreate(&str3);

// 	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img_test.txt", 1);
// 	//gs = (int)ceil((float) (M/3) / bs);   

// 	cudaMemcpyAsync(&f_in[size*0], &f_host[size*0], size * sizeof(float), cudaMemcpyHostToDevice, str1);
// 	cudaMemcpyAsync(&f_in[size*1], &f_host[size*1], size * sizeof(float), cudaMemcpyHostToDevice, str2);
// 	cudaMemcpyAsync(&f_in[size*2], &f_host[size*2], size * sizeof(float), cudaMemcpyHostToDevice, str3); 

// 	//kernel calls
// 	cudaEventCreate(&ct1);
// 	cudaEventCreate(&ct2);
// 	cudaEventRecord(ct1);

// 	//convolucion
// 	//juntar canales
// 	//max pooling

// 	cudaEventRecord(ct2);
// 	cudaEventSynchronize(ct2);
// 	cudaEventElapsedTime(&dt, ct1, ct2);

// 	//solo se copia un canal al host, ya que los 3 canales son iguales
// 	cudaMemcpyAsync(&out[size*0], &f_in[size*0], size * sizeof(float), cudaMemcpyDeviceToHost,str1);

// 	cudaDeviceSynchronize();
// 	//Write(out, M, N, "initial_S.txt\0");
// 	//imprimir_malla(out, N,M);
// 	std::cout << "Tiempo " << ": " << dt << "[ms]" << std::endl;
// 	cudaFree(f_host);
// 	cudaFree(f_in);
// 	cudaFree(f_out);
// }
int main(int argc, char **argv){

	/*
     *  Parte CPU
     */
	cnn_CPU();

	/*
	 *  Parte GPU
	 */

	AoS_GPU(); // Con Memoria Global
	AoS_MC_GPU(); // Con Memoria Compartida
	
	/*
	 *  Memoria Global
	 */
	

	/*
	 *  Memoria Compartida
	 */

	/*
	 * Streams
	 */




 //    cudaFree(R); cudaFree(G); cudaFree(B);
	// cudaFree(Rout); cudaFree(Gout); cudaFree(Bout);
	return 0;
}