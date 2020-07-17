#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// Variables globales GPU y CPU
#define l_kernel 2

/******************************
 *  Procesamiento Matriz CPU  *
 ******************************/

/*
 *  Lectura Archivo
*/
void Read(float** R, float** G, float** B, int *M, int *N, const char *filename, int tipo) {    
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", M, N);

    int imsize = (*M) * (*N);
    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];

    if (tipo == 0){ // Lectura normal
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
 *  Generador de float array kernel
 */
void GenMatrix(float** kernel, int N, int M) {
    float *kernel1 = new float[M*N];
	for(int i = 0; i < M*N; i++){
		// Random entre -5 y 5
		kernel1[i] = ((rand() % 5000) /1000.0) - (rand() % 3);
	}
    *kernel = kernel1;
}

/*
 *  Imprimir Array como matriz
 */
void ShowMatrix(float *matrix, int N, int M) {
    for(int i = 0; i < N; i++){
    	for(int j = 0; j < M; j++)
    		printf("%.1f ", matrix[j + i*M]);
    	printf("\n");
    }
    printf("\n");
}

/*
 *  "Producto" Matricial sub_A * kernel = C
 *  id: id del primer elemento de la submatriz, N: ancho matriz R,G,B
 */
float Product_Matrix(float *A, float *B, int N, int id){
	int col, row, idx_kernel;
	float count;
	col = id%N;
	row = id/N;
	count = 0.0;
	// Recorremos stride
	idx_kernel = 0;
	for(int i=row; i < row+l_kernel; i++){
		for(int j=col; j< col+l_kernel; j++){
			int id_casilla = j + i*N;
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
void ConvolucionCPU(float *A, float **out, float *kernel, int M, int N, int id_kernel){
	float* temp = new float[N*M];
	for(int i=0; i < M; i++){
		for(int j=0; j < N; j++){
			int id = j + i*(N + l_kernel - 1);
			// printf("id: %d\n", id);
			int new_id = j + i*N;
			temp[new_id] = Product_Matrix(A, kernel, N + l_kernel - 1, id);
		}
	}
	*out = temp;
}

/*
 *  Suma de Matrices R,G,B y Funcion de activacion RELU
 */
void SumaMatrizCPU(float **out, float *R, float *G, float *B, int M, int N){
	float* sum = new float[M*N];
	for(int i=0; i < M*N; i++){
		sum[i] = R[i] + G[i] + B[i] > 0 ? R[i] + G[i] + B[i] : 0;
	}
	*out = sum;
}

/*
 *  Funcion de activacion RELU
 */
void ReluCPU(float **out, int M, int N){
	float* sum = new float[M*N];
	for(int i=0; i < M*N; i++){
		sum[i] = (*out)[i] > 0 ? (*out)[i] : 0;
	}
	*out = sum;
}

/*
 *  Procesamiento GPU
 */

/*
 *  Codigo Principal
 */
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	int M, N, M_initial, N_initial;
	const int N_kernels = 20;
	float *kernels[N_kernels]; // Conjunto de kernels(matrices) a usar
	float *output_images[N_kernels]; // Conjunto de imagenes(matrices) de salida por kernel
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout, *Ghostout, *Bhostout;
    // float *R, *G, *B;
    // float *Rout, *Gout, *Bout;
	// int gs = 1, bs = 1024;
	// float dt;
	// cudaEvent_t ct1, ct2;

    // Lectura de archivo
	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img_test.txt", 0);

	// Generar kernels
    for(int j=0; j<N_kernels; j++){
    	GenMatrix(&kernels[j], l_kernel, l_kernel);
    }

	/*
     *  Parte CPU
     */

    // Por cada proceso de convolucion
    for(int c=0; c<5; c++){
    	printf("########## Convolucion %d ###########\n\n", c+1);
    	// Se utiliza el ultimo M,N
    	M_initial = M; N_initial = N;
    	// Por cada kernel a utilizar
		for(int k=0; k<2; k++){
			// Se restablecen valores originales para un nuevo kernel
			M = M_initial; N = N_initial;
			int id_kernel = k;
			printf("Kernel %d:\n", id_kernel);
			ShowMatrix(kernels[id_kernel], l_kernel, l_kernel);

			printf("M: %d N: %d\n", M, N);
			// Actualizamos N,M si aun se puede
			if(N - l_kernel + 1 > 0 && M - l_kernel + 1 > 0){
				N = N - l_kernel + 1;
				M = M - l_kernel + 1;
			} else{
				continue;
			}
			// Si es el primero se suman las matrices RGB resultantes
			if(c == 0){
				// Convoluciones y suma de RGB
				printf("Matriz Rhost:\n");
				ShowMatrix(Rhost, M + l_kernel - 1, N + l_kernel - 1);
				printf("Matriz Ghost:\n");
				ShowMatrix(Ghost, M + l_kernel - 1, N + l_kernel - 1);
				printf("Matriz Bhost:\n");
				ShowMatrix(Bhost, M + l_kernel - 1, N + l_kernel - 1);
				printf("M_out: %d N_out: %d\n", M, N);
				ConvolucionCPU(Rhost, &Rhostout, kernels[id_kernel], M, N, id_kernel);
				ConvolucionCPU(Ghost, &Ghostout, kernels[id_kernel], M, N, id_kernel);
				ConvolucionCPU(Bhost, &Bhostout, kernels[id_kernel], M, N, id_kernel);
				SumaMatrizCPU(&output_images[id_kernel], Rhostout, Ghostout, Bhostout, M, N);
			} else {
				// Convolucion
				printf("Matriz:\n");
				ShowMatrix(output_images[id_kernel], M+ l_kernel - 1, N+ l_kernel - 1);
				printf("M_out: %d N_out: %d\n", M, N);
				ConvolucionCPU(output_images[id_kernel], &output_images[id_kernel], kernels[id_kernel], M, N, id_kernel);
				ReluCPU(&output_images[id_kernel], M, N);
			}
			printf("Imagen salida %d:\n", c);
			ShowMatrix(output_images[id_kernel], M, N);
		}
		// printf("Imagen salida %d:\n", c);
		// ShowMatrix(output_images[0], M, N);
    }

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


 //    cudaFree(R); cudaFree(G); cudaFree(B);
	// cudaFree(Rout); cudaFree(Gout); cudaFree(Bout);
	delete[] Rhost; delete[] Ghost; delete[] Bhost; free(kernels);
	// delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	return 0;
}