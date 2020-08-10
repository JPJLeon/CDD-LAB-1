#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

// Variables globales GPU y CPU
#define l_kernel 3

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
	if (tipo != 1){
    	float* R1 = new float[imsize];
    	float* G1 = new float[imsize];
    	float* B1 = new float[imsize];
	}

    if (tipo == 0){ // Lectura normal
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
    		printf("%.1f ", matrix[j + i*M]);
    	printf("\n");
    }
    printf("\n");
}

/*
 *  "Producto" Matricial sub_A * kernel = C
 *  id: id del primer elemento de la submatriz, N: ancho matriz R
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
			// id del primer elemento de la submatriz
			int id = j + i*3*N;
			// printf("id: %d\n", id);
			int new_id = j + i*N;
			temp[new_id] = Product_Matrix(A, kernel, N*3, id);
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
		sum[i] = R[i]+G[i]+B[i];
	}
	*out = sum;
}

/*
 *  Funcion de activacion RELU
 */
void ReluCPU(float **out, int M, int N){
	float* sum = new float[M*N];
	for(int i=0; i < M*N; i++){
		sum[i] = MaxCPU((*out)[i], 0.0);
	}
	*out = sum;
}

/*
 *  Funcion Max pooling 2x2
 */
void PoolingCPU(float **out, int *M, int *N){
	int new_N, new_M;
	float max, v1, v2, v3, v4;
	new_N = (*N)/2;
	new_M = (*M)/2;
	if((*N)%2){
		new_N++;
	}
	if((*M)%2){
		new_M++;
	}
	// printf("new_M: %d new_N: %d\n", new_M, new_N);
	float* temp = new float[new_N*new_M];
	for(int i=0; i < new_M; i++){
		for(int j=0; j < new_N; j++){
			v1 = (*out)[j*2 + i*2*(*N)];
			// printf("v1: %d\n", j*2 + i*2*(*N));
			v2 = 0;
			v3 = 0;
			v4 = 0;
			// Agregamos los valores extremos en caso de ser N o M impar
			if(j != new_N-1){
				v2 = (*out)[j*2 + 1 + i*2*(*N)];
				if(new_M == 1 || i != new_M-1){
					v4 = (*out)[j*2 + 1 + (i+1)*(*N)];
				}
			}
			if(new_M == 1 || i != new_M-1){
				v3 = (*out)[j*2 + (i+1)*(*N)];
			}
			max = MaxCPU(MaxCPU(v1, v2), MaxCPU(v3, v4));
			temp[j + i*new_N] = max;
		}
	}
	*out = temp;
	*N = new_N; *M = new_M;
}

void cnn_CPU(float *Rhost, float *Ghost, float *Rhostout, float *Ghostout, float *Bhostout, float *Bhost, float *kernel, int M, int N){
	float *output_image = new float[M*N]; // Conjunto de imagenes(matrices) de salida por kernel
	printf("Matriz original: %d x %d\n", M, N);
	// ShowMatrix(Rhost, M, N);
	// Por cada proceso de convolucion
    for(int c=0; c<3; c++){
    	printf("\n########## Convolucion %d ###########\n", c+1);
		// Actualizamos N,M si aun se puede
		if(N/3 > 0 && M/3 > 0){
			printf("M: %d N: %d\n", M, N);
			N = N/3;
			M = M/3;
		} else{
			continue;
		}
		// Si es el primero se suman las matrices RGB resultantes
		if(c == 0){
			// ShowMatrix(Rhost, 3*M, 3*N);
			ConvolucionCPU(Rhost, &Rhostout, kernel, M, N, 0);
			ConvolucionCPU(Ghost, &Ghostout, kernel, M, N, 0);
			ConvolucionCPU(Bhost, &Bhostout, kernel, M, N, 0);
			SumaMatrizCPU(&output_image, Rhost, Ghost, Bhost, M, N);
		} else {
			// ShowMatrix(output_image, 3*M, 3*N);
			ConvolucionCPU(output_image, &output_image, kernel, M, N, 0);
		}
		// ReluCPU(&output_image, M, N);
		printf("Matriz Convolucion %d: %d x %d\n", c+1, M, N);
		// ShowMatrix(output_image, M, N);
		if(M*N > 3){
			PoolingCPU(&output_image, &M, &N);
			printf("Imagen pooling %d: %d x %d\n", c+1, M, N);
			// ShowMatrix(output_image, M, N);
		}
    }
    printf("Imagen salida: %d x %d\n", M, N);
	ShowMatrix(output_image, M, N);
}

/*
 *  Procesamiento GPU
 */

/*
 *  Codigo Principal
 */

void streams (){
	//procesamiento de las convoluciones con 3 streams, 1 por cada color
	cudaEvent_t ct1, ct2;	
	float dt;

	int M, N;
	int gs, bs = 256;

	float kernel[l_kernel*l_kernel] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Conjunto de kernel(matrices) a usar
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout;

    Rhostout = new float[l_kernel*l_kernel];

	//crear streams
	cudaStream_t str1, str2, str3;
	cudaStreamCreate(&str1);
	cudaStreamCreate(&str2);
	cudaStreamCreate(&str3);

	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img_test.txt", 1);
	//gs = (int)ceil((float) (M/3) / bs);   

	cudaMemcpyAsync(&f_in[size*0], &f_host[size*0], size * sizeof(float), cudaMemcpyHostToDevice, str1);
	cudaMemcpyAsync(&f_in[size*1], &f_host[size*1], size * sizeof(float), cudaMemcpyHostToDevice, str2);
	cudaMemcpyAsync(&f_in[size*2], &f_host[size*2], size * sizeof(float), cudaMemcpyHostToDevice, str3); 

	//kernel calls
	cudaEventCreate(&ct1);
	cudaEventCreate(&ct2);
	cudaEventRecord(ct1);

	//convolucion
	//juntar canales
	//max pooling

	cudaEventRecord(ct2);
	cudaEventSynchronize(ct2);
	cudaEventElapsedTime(&dt, ct1, ct2);

	//solo se copia un canal al host, ya que los 3 canales son iguales
	cudaMemcpyAsync(&out[size*0], &f_in[size*0], size * sizeof(float), cudaMemcpyDeviceToHost,str1);

	cudaDeviceSynchronize();
	//Write(out, M, N, "initial_S.txt\0");
	//imprimir_malla(out, N,M);
	std::cout << "Tiempo " << ": " << dt << "[ms]" << std::endl;
	cudaFree(f_host);
	cudaFree(f_in);
	cudaFree(f_out);
}
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	int M, N;
	float kernel[l_kernel*l_kernel] = {0, 1, 0, 1, -4, 1, 0, 1, 0}; // Conjunto de kernel(matrices) a usar
	// float *kernel = new float[l_kernel*l_kernel];
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout, *Ghostout, *Bhostout;

    Rhostout = new float[l_kernel*l_kernel];
    Ghostout = new float[l_kernel*l_kernel];
    Bhostout = new float[l_kernel*l_kernel];

    // float *R, *G, *B;
    // float *Rout, *Gout, *Bout;
	// int gs = 1, bs = 1024;
	// float dt;
	// cudaEvent_t ct1, ct2;

    // Lectura de archivo
	Read(&Rhost, &Ghost, &Bhost, &M, &N, "img.txt", 0);
	// kernel = &array[0];
	printf("Kernel:\n");
	ShowMatrix(kernel, l_kernel, l_kernel);

	/*
     *  Parte CPU
     */
	cnn_CPU(Rhost, Ghost, Rhostout, Ghostout, Bhostout, Bhost, kernel, M, N);

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
	delete[] Rhost; delete[] Ghost; delete[] Bhost;
	delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	return 0;
}