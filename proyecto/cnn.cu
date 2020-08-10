#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// Variables globales GPU y CPU
#define l_kernel 3
#define stride 1

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
	else if (tipo == 1){ //lectura en pinned memory
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
    		printf("%.2f ", matrix[j + i*M]);
    	printf("\n");
    }
    printf("\n");
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
void ConvolucionCPU(float *A, float **out, float *kernel, int M, int N){
	float* temp = new float[N*M];
	int col, row, id;
	int N_original;
	if(stride == 1){
		N_original = N + l_kernel - 1;
	} else{
		N_original = N*stride;
	}

	int count_output = 0;
	for(int i=0; i < N_original*N_original; i=i+stride){
		// printf("id: %d\n", id);
		if(i%N_original < N && i/N_original < N){
			// printf("i: %d out:%d\n", i, count_output);
			temp[count_output] = Product_Matrix(A, kernel, N_original, i);
			count_output++;
		}
	}
	*out = temp;
}

// 1 1 1 1 1 1 1 1
// 1 1 1 1 1 1 1 1
// 1 1 1 1 1 1 1 1
// 1 1 1 1 1 1 1 1
// 1 1 1 1 1 1 1 1
// 1 1 1 1 1 1 1 1
// 1 1 1 1 1 1 1 1
// 1 1 1 1 1 1 1 1

// 1 1 1 1 1 1
// 1 1 1 1 1 1
// 1 1 1 1 1 1
// 1 1 1 1 1 1
// 1 1 1 1 1 1
// 1 1 1 1 1 1

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
			v1 = (*out)[j*2 + i*2*(*N)];
			// printf("v1: %d\n", j*2 + i*2*(*N));
			v2 = (*out)[j*2 + 1 + i*2*(*N)];
			v3 = (*out)[j*2 + (i+1)*(*N)];
			v4 = (*out)[j*2 + 1 + (i+1)*(*N)];
			temp[j + i*new_N] = MaxCPU(MaxCPU(v1, v2), MaxCPU(v3, v4));
		}
	}
	*out = temp;
	*N = new_N; *M = new_M;
}

void cnn_CPU(){
	int M, N;
	float array[l_kernel*l_kernel] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Conjunto de kernel(matrices) a usar
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
	// ShowMatrix(Rhost, M, N);
	// Por cada proceso de convolucion
    for(int c=0; c<1; c++){
    	printf("\n########## Convolucion %d ###########\n", c+1);
		// Actualizamos N,M si aun se puede
		// if(N/3 > 0 && M/3 > 0){
			printf("M: %d N: %d\n", M, N);
			if(stride == 1){
				N = N - l_kernel + 1;
				M = M - l_kernel + 1;
			} else{
				N = N/stride;
				M = M/stride;
			}
		// } else{
		// 	continue;
		// }
		// Si es el primero se suman las matrices RGB resultantes
		if(c == 0){
			// ShowMatrix(Rhost, M + l_kernel -1, N + l_kernel -1);
			ConvolucionCPU(Rhost, &Rhostout, kernel, M, N);
			ConvolucionCPU(Ghost, &Ghostout, kernel, M, N);
			ConvolucionCPU(Bhost, &Bhostout, kernel, M, N);
			// ShowMatrix(Rhostout, M, N);
			// ShowMatrix(Ghostout, M, N);
			// ShowMatrix(Bhostout, M, N);
			SumaMatrizCPU(&output_image, Rhostout, Ghostout, Bhostout, M, N);
		} else {
			// ShowMatrix(output_image, stride*M, stride*N);
			ConvolucionCPU(output_image, &output_image, kernel, M, N);
		}
		printf("Matriz Convolucion %d: %d x %d\n", c+1, M, N);
		// ShowMatrix(output_image, M, N);
    }

 //    PoolingCPU(&output_image, &M, &N);
	// printf("Imagen pooling: %d x %d\n", M, N);
	// ShowMatrix(output_image, M, N);
	ReluCPU(output_image, M, N);

    printf("Imagen salida: %d x %d\n", M, N);
	// ShowMatrix(output_image, M, N);
	Write(output_image, M, N, "ResultadoCPU.txt");
	delete[] Rhost; delete[] Ghost; delete[] Bhost;
	delete[] Rhostout; delete[] Ghostout; delete[] Bhostout, delete[] output_image;
}

/*
 *  Procesamiento GPU
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

    // gs = (int)ceil((float) Mres*Nres / bs);    
    // cudaMalloc((void**)&R, M * N * sizeof(float));
    // cudaMalloc((void**)&G, M * N * sizeof(float));
    // cudaMalloc((void**)&B, M * N * sizeof(float));
    // cudaMemcpy(R, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(G, Ghost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(B, Bhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
        
    // cudaMalloc((void**)&Rout, M * N * sizeof(float));
    // cudaMalloc((void**)&Gout, M * N * sizeof(float));
    // cudaMalloc((void**)&Bout, M * N * sizeof(float));
    
    // cudaEventCreate(&ct1);
    // cudaEventCreate(&ct2);
    // cudaEventRecord(ct1);
    // kernel<<<gs, bs>>>(R, G, B, Rout, Gout, Bout, Mres, Nres, X, Y, N);
    // cudaEventRecord(ct2);
    // cudaEventSynchronize(ct2);
    // cudaEventElapsedTime(&dt, ct1, ct2);
    // std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;


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