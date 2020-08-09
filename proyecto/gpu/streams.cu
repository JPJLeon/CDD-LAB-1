#include <iostream>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// Variables globales GPU y CPU
#define l_kernel 3
#define stride 3

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
void Write(float* R, float* G, float* B, 
	int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", M, N);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", R[i]);
    fprintf(fp, "%f\n", R[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", G[i]);
    fprintf(fp, "%f\n", G[M*N-1]);
    for(int i = 0; i < M*N-1; i++)
        fprintf(fp, "%f ", B[i]);
    fprintf(fp, "%f\n", B[M*N-1]);
    fclose(fp);
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

__global__ void convolucion(float *f, float *f_out ,float* kernel, int N, int Nres){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int x,y; 
	if(tid < 800*800){ //1 thread para cada pixel de salida
	  x = 1 +  (tid%Nres)*stride; //coordenaas del centro de cada sub_matriz
		y = 1 + (tid/Nres)*stride;

		float suma = 0;
		int indice_sub_matriz, indice_kernel;
		for (int i = -1; i<=1 ; i++){
				for (int j = -1; j <= 1; j++){
						indice_sub_matriz = (x+i) + (y+j)*N;
						indice_kernel = (1+i) + (1+j)*3;
						suma += f[indice_sub_matriz] * kernel[indice_kernel];
				}
		}
		printf("%f\n", suma);
	  f_out[tid] = suma;
	}
}


__device__ float max_pool(float *f, int N, int x, int y){
		//recorre una sub matriz de 2x2 y encuentra el valor máximo 
		float valores[4] = {
				f[(x+0) + (y+0)*N],
				f[(x+1) + (y+0)*N],
				f[(x+0) + (y+1)*N],
				f[(x+1) + (y+1)*N]
		};
		int max = 0;
		for (int i = 0; i< 4; i++){
				if (valores[i] > max){
						max = valores[i];  
				}
		}
		return max;
}
__global__ void pooling(float *f, float *f_out, int N){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int x,y;
	if(tid < 400*400){ //1 thread para cada pixel de salida
	  x = 1 +  (tid%N)*stride;
		y = 1 + (tid/N)*stride;
	  f_out[tid] = max_pool(f, N , x, y);
	}
}


/*
 *  Codigo Principal
 */
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	int M, N;
	
	float kernel[l_kernel*l_kernel] = {-1, 1, 0, -1, 1, 0 ,-1, 1, 0}; // filtro a usar
  float *Rhost, *Ghost, *Bhost;
  float *Rhostout, *Ghostout, *Bhostout;

  float *R, *G, *B;
  float *Rout, *Gout, *Bout;
	int gs, bs = 256;
  float dt;
  cudaEvent_t ct1, ct2;

  // Lectura de archivo
	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img.txt", 0);
	int Mres = M/l_kernel;
	int Nres = N/l_kernel;
	Rhostout = new float[Mres*Nres];
	Ghostout = new float[Mres*Nres];
	Bhostout = new float[Mres*Nres];
	
	/*
	 *  Parte GPU
	 */

    gs = (int)ceil((float) Mres*Nres / bs);    
    cudaMalloc((void**)&R, M * N * sizeof(float));
    cudaMalloc((void**)&G, M * N * sizeof(float));
    cudaMalloc((void**)&B, M * N * sizeof(float));
    cudaMemcpy(R, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(G, Ghost, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B, Bhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
        
    cudaMalloc((void**)&Rout, Mres * Nres * sizeof(float));
    cudaMalloc((void**)&Gout, Mres * Nres * sizeof(float));
    cudaMalloc((void**)&Bout, Mres * Nres * sizeof(float));
    
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    convolucion<<<gs, bs>>>(R, Rout, kernel, N, Nres);
		convolucion<<<gs, bs>>>(G, Gout, kernel, N, Nres);
		convolucion<<<gs, bs>>>(B, Bout, kernel, N, Nres);

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);

		cudaMemcpy(Rhostout, Rout, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Ghostout, Gout, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(Bhostout, Bout, M * N * sizeof(float), cudaMemcpyHostToDevice);

		std::cout << "Tiempo: " << dt << "[ms]" << std::endl;

		Write(Rhostout, Ghostout, Bhostout, Mres, Nres, "resultado.txt");


	/*
	 *  Memoria Global
	 */
	

	/*
	 *  Memoria Compartida
	 */

	cudaFree(R); cudaFree(G); cudaFree(B);
	cudaFree(Rout); cudaFree(Gout); cudaFree(Bout);
	delete[] Rhost; delete[] Ghost; delete[] Bhost;
	delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	return 0;
}