// Basado en el codigo entregado durante clases practicas

#include <iostream>
#include <time.h>
#include <cuda_runtime.h>

/*
 *  Lectura Archivo
 */
void Read(float** R, float** G, float** B, int *N, int *S, int** positions, const char *filename) {
    printf("Leemos el archivo %s!\n", filename); 
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", N, S);
    // obtenemos segunda linea con nuevas posiciones
    int P = (*N) / (*S);
    int *positions1 = new int[P*P];
    for(int i = 0; i < P*P; i++){
	    fscanf(fp, "%*d ", &positions[i]);
    }
	
    
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


__host__ __device__ int findNewPosition(int P, int position, int* positions){

	int newPosition;

	for (int j = 0; j < P*P; j++) {
    	if(position == positions[j]) {
        	newPosition = j;
        	break;
    	}
	}

	return newPosition;
}

/*
 *  Procesamiento Imagen CPU
 */
void funcionCPU( float* R, float* G, float* B, float* Rout, float* Gout, float* Bout, int N, int S, int* positions){
	int P = N/S;
	//cambiar
	for (int i = 0; i < N*N; i++){
		int x = i % N;
		int y = i / N;
		int newX = x / P;
		int newY = y / P;
		int position = newX + newY * P;
		int newPosition = findNewPosition(P, position, positions);
		int newI;

		newI = (i + S*S*(newPosition - position));

		Rout[newI] = R[i];
		Gout[newI] = G[i];
		Bout[newI] = B[i];
	}
}

/*
 *  Procesamiento Imagen GPU
 */
__global__ void kernelGPU( float* R, float* G, float* B, float* Rout, float* Gout, float* Bout, int N, int S, int* positions){
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int P = N/S;

	if(i < N*N){
		int x = i % N;
		int y = i / N;
		int newX = x / P;
		int newY = y / P;
		int position = newX + newY * P;
		int newPosition = findNewPosition(P, position, positions);
		int newI;

		newI = (i + S*S*(newPosition - position));

		Rout[newI] = R[i];
		Gout[newI] = G[i];
		Bout[newI] = B[i];
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
    char names[5][3][30] = {{"img100x100.txt\0", "img100x100CPU.txt\0", "img100x100GPU.txt\0"}, 
    	{"img200x200.txt\0", "img200x200CPU.txt\0", "img200x200GPU.txt\0"},
    	{"img400x400.txt\0", "img400x400CPU.txt\0", "img400x400GPU.txt\0"},
    	{"img800x800.txt\0", "img800x800CPU.txt\0", "img800x800GPU.txt\0"},
    	{"img1600x1600.txt\0", "img1600x1600CPU.txt\0", "img1600x1600GPU.txt\0"}};

    for (int i=0; i<5; i++){
	    Read(&Rhost, &Ghost, &Bhost, &N, &S, &positions, names[i][0]); //leemos archivo y reservamos memoria

	    /*
	     *  Parte CPU
	     */
	    Rhostout = new float[N*N];	//reservamos memoria
	    Ghostout = new float[N*N];
	    Bhostout = new float[N*N];

	    t1 = clock();
	    funcionCPU(Rhost, Ghost, Bhost, Rhostout, Ghostout, Bhostout, N, S, positions); // Agregar parametros!
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
	    cudaMemcpy(Rdev, Rhost, N * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Gdev, Ghost, N * N * sizeof(float), cudaMemcpyHostToDevice);
	    cudaMemcpy(Bdev, Bhost, N * N * sizeof(float), cudaMemcpyHostToDevice);
	        
	    cudaMalloc((void**)&Rdevout, N * N * sizeof(float));
	    cudaMalloc((void**)&Gdevout, N * N * sizeof(float));
	    cudaMalloc((void**)&Bdevout, N * N * sizeof(float));
	    
	    cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);
	    kernelGPU<<<grid_size, block_size>>>(Rdev, Gdev, Bdev, Rdevout, Gdevout, Bdevout, N, S, positions); // Agregar parametros!
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
    	delete[] Rhost; delete[] Ghost; delete[] Bhost;
    	delete[] Rhostout; delete[] Ghostout; delete[] Bhostout;
	}
	return 0;
}