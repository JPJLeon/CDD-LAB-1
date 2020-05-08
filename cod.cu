// Basado en el codigo entregado durante clases practicas

#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

/*
 *  Lectura Archivo
 */
void Read(float** R, float** G, float** B, int** positions, int *N, int *S, const char *filename) {    
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", N, S);
    
    // obtenemos segunda linea con nuevas posiciones
    int P = (*N) / (*S);
    int *positions1 = new int[P*P];
    for(int i = 0; i < P*P; i++)
	    fscanf(fp, "%d ", &(positions[i]));
	
    int imsize = (*N) * (*N);
    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(R1[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(G1[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(B1[i]));
    fclose(fp);

    *R = R1; *G = G1; *B = B1; *positions = positions1;
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
 *  Procesamiento Imagen CPU
 */
void funcionCPU( float* R, float* G, float* B, float* Rout, float* Gout, float* Bout, int M, int N ){

	//cambiar
	for (int i = 0; i < N*M; i++){
		Rout[i] = 1 - R[i];
		Gout[i] = 1 - G[i];
		Bout[i] = 1 - B[i];
	}
}

/*
 *  Procesamiento Imagen GPU
 */
__global__ void kernelGPU( float* R, float* G, float* B, float* Rout, float* Gout, float* Bout, int M, int N){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;

	// cambiar
	if(tid < M*N){
		Rout[tid] = 1 - R[tid];
		Gout[tid] = 1 - G[tid];
		Bout[tid] = 1 - B[tid];
	} 
}

/*
 *  Codigo Principal
 */
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	clock_t t1, t2;
	cudaEvent_t ct1, ct2;
	double ms;
	float dt;
	int N, S;
	int *positions;
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout, *Ghostout, *Bhostout;
    float *Rdev, *Gdev, *Bdev;
    float *Rdevout, *Gdevout, *Bdevout;
    
    //cambiar con los nombres de los archivos correctos
    //char names[2][3][15] = {{"img.txt\0", "imgPCPU.txt\0", "imgPGPU.txt\0"}, {"imgG.txt\0", "imgGCPU.txt\0", "imgGGPU.txt\0"}};

    for (int i=0; i<2; i++){
	    Read(&Rhost, &Ghost, &Bhost, &N, &S, &positions, names[i][0]); //leemos archivo y reservamos memoria

	    /*
	     *  Parte CPU
	     */
	    Rhostout = new float[M*N];	//reservamos memoria
	    Ghostout = new float[M*N];
	    Bhostout = new float[M*N];

	    t1 = clock();
	    //funcionCPU(Rhost, Ghost, Bhost, Rhostout, Ghostout, Bhostout, M, N); // Agregar parametros!
	    t2 = clock();
	    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	    std::cout << "Tiempo CPU: " << ms << "[ms]" << std::endl;
	    Write(Rhostout, Ghostout, Bhostout, M, N, names[i][1]);

	    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	    
	    /*
	     *  Parte GPU
	     */

	    int grid_size, block_size = 256;
	    grid_size = (int)ceil((float) M * N / block_size);
	        
	    cudaMalloc((void**)&Rdev, M * N * sizeof(float));
	    cudaMalloc((void**)&Gdev, M * N * sizeof(float));
	    cudaMalloc((void**)&Bdev, M * N * sizeof(float));
	    cudaMemcpy(Rdev, Rhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Gdev, Ghost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Bdev, Bhost, M * N * sizeof(float), cudaMemcpyHostToDevice);
	        
	    cudaMalloc((void**)&Rdevout, M * N * sizeof(float));
	    cudaMalloc((void**)&Gdevout, M * N * sizeof(float));
	    cudaMalloc((void**)&Bdevout, M * N * sizeof(float));
	    
	    cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);
	    //kernelGPU<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout, Gdevout, Bdevout, M, N); // Agregar parametros!
	    cudaEventRecord(ct2);
	    cudaEventSynchronize(ct2);
	    cudaEventElapsedTime(&dt, ct1, ct2);
	    std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;

	    Rhostout = new float[M*N];
	    Ghostout = new float[M*N];
	    Bhostout = new float[M*N];
	    cudaMemcpy(Rhostout, Rdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Ghostout, Gdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Bhostout, Bdevout, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	    Write(Rhostout, Ghostout, Bhostout, M, N, names[i][2]);

	    cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);
    	cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
    	delete[] Rhost; delete[] Ghost; delete[] Bhost;
    	delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	}
	return 0;
}