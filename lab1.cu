#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

/*
 *  Lectura Archivo
 */
void Read(float** R, float** G, float** B, 
	int *N, int *S, int **ordenamiento, int* P, const char *filename) {    
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", N, S);

    int imsize = (*N) * (*N);
    *P = (*N)/ (*S);
    int P2 = (*P) * (*P) ;

    float* R1 = new float[imsize];
    float* G1 = new float[imsize];
    float* B1 = new float[imsize];
    int *orden_temp = new int[P2];

    for(int i = 0; i < P2; i++)
	    fscanf(fp, "%d ", &(orden_temp[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(R1[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(G1[i]));
	for(int i = 0; i < imsize; i++)
	    fscanf(fp, "%f ", &(B1[i]));
    fclose(fp);
    *R = R1; *G = G1; *B = B1;
    *ordenamiento = orden_temp;
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
void funcionCPU(float* R,float* G,float* B, float* Rout,float* Gout,float* Bout, int N, int S, int P, int* ordenamiento){
	for (int i = 0; i< N*N; i++){
        int x = i % N;
        int y = i / N;

        int x_bloque_escritura = x / S;
        int y_bloque_escritura = y /S;

        int indice_bloque_escritura = x_bloque_escritura + y_bloque_escritura * P;
        int indice_bloque_lectura = ordenamiento[indice_bloque_escritura];

        int x_bloque_lectura = indice_bloque_lectura % P;
        int y_bloque_lectura = indice_bloque_lectura / P;

        int shift_horizontal = x_bloque_lectura - x_bloque_escritura;
        int shift_vertical = y_bloque_lectura - y_bloque_escritura;

        x = x + shift_horizontal * S;
        y = y + shift_vertical * S;

        Rout[i] = R[x + y*N];
        Gout[i] = G[x + y*N];
        Bout[i] = B[x + y*N]; 

  }
}

/*
 *  Procesamiento Imagen GPU
 */
__global__ void kernelGPU(float* R,float* G,float* B, float* Rout,float* Gout,float* Bout, int N, int S, int P, int* ordenamiento){
  
	int i = threadIdx.x + blockDim.x* blockIdx.x;
	if (i < N*N){
		int x = i % N;
        int y = i / N;

        int x_bloque_escritura = x / S;
        int y_bloque_escritura = y /S;

        int indice_bloque_escritura = x_bloque_escritura + y_bloque_escritura * P;
        int indice_bloque_lectura = ordenamiento[indice_bloque_escritura];

        int x_bloque_lectura = indice_bloque_lectura % P;
        int y_bloque_lectura = indice_bloque_lectura / P;

        int shift_horizontal = x_bloque_lectura - x_bloque_escritura;
        int shift_vertical = y_bloque_lectura - y_bloque_escritura;

        x = x + shift_horizontal * S;
        y = y + shift_vertical * S;

        Rout[i] = R[x + y*N];
        Gout[i] = G[x + y*N];
        Bout[i] = B[x + y*N]; 
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
    int P;
    int *ordenamiento;   //arreglo con el ordenamiento
    int* ordenamiento_dev;
    float *Rhost, *Ghost, *Bhost;
    float *Rhostout, *Ghostout, *Bhostout;
    float *Rdev, *Gdev, *Bdev;
    float *Rdevout, *Gdevout, *Bdevout;
    char names[1][3][20] = {{"img100x100.txt\0", "img100x100CPU.txt\0", "img100x100GPU.txt\0"}};

    for (int i=0; i<1; i++){
	    Read(&Rhost, &Ghost, &Bhost, &N, &S, &ordenamiento, &P,  names[i][0]);


	    /*
	     *  Parte CPU
	     */
	    Rhostout = new float[N*N];
	    Ghostout = new float[N*N];
	    Bhostout = new float[N*N];

	    t1 = clock();
	    funcionCPU(Rhost,Ghost,Bhost, Rhostout,Ghostout,Bhostout, N, S, P, ordenamiento); // Agregar parametros!
	    t2 = clock();
	    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
	    std::cout << "Tiempo CPU: " << ms << "[ms]" << std::endl;
	    Write(Rhostout, Ghostout, Bhostout, N, N, names[i][1]);

	    delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	    
	    /*
	     *  Parte GPU
	     */
	    int grid_size, block_size = 256;
	    grid_size = (int)ceil((float) N * N / block_size);
	        
	    cudaMalloc((void**)&Rdev, N * N * sizeof(float));
	    cudaMalloc((void**)&Gdev, N * N * sizeof(float));
	    cudaMalloc((void**)&Bdev, N * N * sizeof(float));
        cudaMalloc((void**)&ordenamiento_dev, (P*P) * sizeof(int));
	    cudaMemcpy(Rdev, Rhost, N * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Gdev, Ghost, N * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Bdev, Bhost, N * N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ordenamiento_dev, ordenamiento, (P*P) * sizeof(int), cudaMemcpyHostToDevice);
	        
	    cudaMalloc((void**)&Rdevout, N * N * sizeof(float));
	    cudaMalloc((void**)&Gdevout, N * N * sizeof(float));
	    cudaMalloc((void**)&Bdevout, N * N * sizeof(float));
	    
	    cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);
	    kernelGPU<<<grid_size, block_size>>>(Rdev,Gdev,Bdev, Rdevout,Gdevout,Bdevout, N, S, P,  ordenamiento_dev); // Agregar parametros!
	    cudaEventRecord(ct2);
	    cudaEventSynchronize(ct2);
	    cudaEventElapsedTime(&dt, ct1, ct2);
	    std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;

	    Rhostout = new float[N*N];
	    Ghostout = new float[N*N];
	    Bhostout = new float[N*N];
	    cudaMemcpy(Rhostout, Rdevout, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Ghostout, Gdevout, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	    cudaMemcpy(Bhostout, Bdevout, N * N * sizeof(float), cudaMemcpyDeviceToHost);
	    Write(Rhostout, Ghostout, Bhostout, N, N, names[i][2]);

    	cudaFree(Rdev); cudaFree(Gdev); cudaFree(Bdev);
    	cudaFree(Rdevout); cudaFree(Gdevout); cudaFree(Bdevout);
        cudaFree(ordenamiento_dev);
    	delete[] Rhost; delete[] Ghost; delete[] Bhost;
    	delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
        delete[] ordenamiento;  
	}
	return 0;
}