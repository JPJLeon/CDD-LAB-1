#include <iostream>
#include <cuda_runtime.h>

/*  Lectura Archivo */
void Read(int **f, int *M, int *N, const char *filename, int X, int tipo) {
    FILE *fp;
    fp = fopen(filename, "r");
    fscanf(fp, "%d %d\n", N, M);

    int imsize = (*M) * (*N) * X;
    int* f1 = new int[imsize];
    int Largo = (*M) * (*N);

    if (tipo == 0){ // AoS
		for(int x=0; x<X; x++){
			for(int i = 0; i < Largo; i++){
	        	fscanf(fp, "%d ", &(f1[i*4 + x]));
		        // printf("%d ", i*4 + x);
			}
	    }

	    // Datos M = 6, N = 4

	    //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 											23
	    // 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 											47
	    // 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 											71
	    // 72 73 74 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 											95
	    
	    //  0 24 48 72  1 25 49 73  2 26 50 74  3 27 51 75  4 28 52 76   5 29 53 77  6 30 54 78  7 31 55 79  8 32 56 80 ... 95

	    //  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1 ..    		 3	(x) < X
	    //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23  0  1  2  3  4  5  6  7  8 ...			23 (i) < N*M

	    //  i*4 + x
	    //  0  4  8 12 16 20 24 28 		....  		1  5  9 13 17 21 25 29    ......	2  6 10 14 18 22 	....	3  7 11 15 19 .. 

	} else{ // SoA 
		for(int j=0; j<X; j++){
	    	for(int i = 0; i < Largo-1; i++){
		        fscanf(fp, "%d ", &(f1[i + j*Largo]));
	    	}
		    fscanf(fp, "%d\n", &(f1[Largo-1 + j*Largo]));
	    }
	}
    fclose(fp);
    *f = f1;
}

/*  Escritura de archivo initial con array */
void Write_AoS(int *f, int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", N, M);
    int Largo = M*N;
    for(int j=0; j<4; j++){
    	for(int i = 0; i < Largo-1; i++){
	        fprintf(fp, "%d ", f[i*4 + j]);
	    	printf("%d ", f[i*4 + j]);
	    }
	    fprintf(fp, "%d\n", f[(Largo-1)*4 + j]);
	    printf("%d\n", f[(Largo-1)*4 + j]);
    }
    printf("\n");
    fclose(fp);
}

/*  Escritura de archivo initial con array */
void Write_SoA(int *f, int M, int N, const char *filename) {
    FILE *fp;
    fp = fopen(filename, "w");
    fprintf(fp, "%d %d\n", N, M);
    int Largo = M*N;
    for(int j=0; j<4; j++){
    	for(int i = 0; i < Largo-1; i++){
	        fprintf(fp, "%d ", f[i]);
	    	printf("%d ", f[i]);
	    }
	    fprintf(fp, "%d\n", f[Largo-1]);
	    printf("%d\n", f[Largo-1]);
    }
    printf("\n");
    fclose(fp);
}

void validar(int *f, int N, int M){
	int suma=0;
	for(int i=0; i<N*M*4; i++){
		suma += f[i];
	}
	printf("Particulas: %d\n", suma);
}

/*  Procesamiento GPU AoS Coalisiones */
__global__ void kernelAoS_col(int *f, int *f_out, int X, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M*N){
		int idb = tid*4;
		int f0, f1, f2, f3;
		// Almacenamos los datos en memoria
		f0 = f[idb];
		f1 = f[idb+1];
		f2 = f[idb+2];
		f3 = f[idb+3];
		if(f0 && f2 && f1 == 0 && f3 == 0){
			f[idb] = 0;
			f[idb+1] = 1;
			f[idb+2] = 0;
			f[idb+3] = 1;
		} else if(f0 == 0 && f2 == 0 && f1 && f3){
			f[idb] = 1;
			f[idb+1] = 0;
			f[idb+2] = 1;
			f[idb+3] = 0;
		}
	}
}

/*  Procesamiento GPU AoS Streaming */
__global__ void kernelAoS_stream(int *f, int *f_out, int j, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N*M){
		// f0: der
		// f1: arr
		// f2: izq
		// f3: abj
		int x, y, idb, borde=0;
		idb = tid*4;
		x = tid % M; // 4
		y = tid / M; // 1
		// Algo que no entiendo pasaba, que se bajaba en el eje y cuando x == 0
		if(x == 0){
			borde = 1;
		}
		// Id de los nodos adyacentes
		int nd[] = { (x+1)%M + (y+borde)*M, 
					x + ((y+1)%N)*M, 
					(x-1)%M + (y+borde)*M, 
					x + ((y-1)%N)*M };
		// Recorremos las direcciones
		for(int i=0; i<4; i++){
			// Seteo todas en 0
			f_out[idb+i] = 0;
			// Si la particula se mueve en esta direccion
			if(f[idb+i] == 1){
				// La direccion del nodo de esa direccion cambia
				f_out[nd[i]*4+i] += 1;
			}
		}
		// Copio todo en f denuevo
		for(int i=0; i<4; i++){
			f[idb+i] = f_out[idb+i];
		}
	}
}

/*  Procesamiento GPU SoA Coalisiones */
__global__ void kernelSoA_col(int *f, int *f_out, int X, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M*N){
		int f0, f1, f2, f3, Largo;
		Largo = N*M;
		// Almacenamos los datos en memoria
		f0 = f[tid];
		f1 = f[tid+1*Largo];
		f2 = f[tid+2*Largo];
		f3 = f[tid+3*Largo];
		if(f0 && f2 && f1 == 0 && f3 == 0){
			f[tid] = 0;
			f[tid+1*Largo] = 1;
			f[tid+2*Largo] = 0;
			f[tid+3*Largo] = 1;
		} else if(f0 == 0 && f2 == 0 && f1 && f3){
			f[tid] = 1;
			f[tid+1*Largo] = 0;
			f[tid+2*Largo] = 1;
			f[tid+3*Largo] = 0;
		}
	}
}

/*  Procesamiento GPU SoA Streaming */
__global__ void kernelSoA_stream(int *f, int *f_out, int X, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N*M){
		// f0: der
		// f1: arr
		// f2: izq
		// f3: abj
		int x, y, borde=0, Largo = N*M;
		x = tid % M; // 4
		y = tid / M; // 1
		// Algo que no entiendo pasaba, que se bajaba en el eje y cuando x == 0
		if(x == 0){
			borde = 1;
		}
		// Id de los nodos adyacentes
		int nd[] = { (x+1)%M + (y+borde)*M,
					x + ((y+1)%N)*M,
					(x-1)%M + (y+borde)*M,
					x + ((y-1)%N)*M };
		// Recorremos las direcciones
		for(int i=0; i<4; i++){
			// Seteo todas en 0
			f_out[tid+1*Largo] = 0;
			// Si la particula se mueve en esta direccion
			if(f[tid+1*Largo] == 1){
				// La direccion del nodo de esa direccion cambia
				f_out[nd[i]*4+i] += 1;
			}
		}
		// Copio todo en f denuevo
		for(int i=0; i<4; i++){
			f[tid+1*Largo] = f_out[tid+1*Largo];
		}
	}
}

/*  Codigo Principal */
int main(int argc, char **argv){

    /*
     *  Inicializacion
     */
	cudaEvent_t ct1, ct2;
	float dt;
	// N eje y, M eje x
	int M, N;
    int *f_host, *f_hostout, *f, *f_out;
    char filename[15] = "initial.txt\0";
	int gs, bs = 256;
	int X = 4;

	// 2 metodos SoA y AoS
    for (int i=0; i<2; i++){
    	Read(&f_host, &M, &N, filename, X, i);

	    gs = (int)ceil((float) M * N * X / bs);    
	    cudaMalloc((void**)&f, M * N * X * sizeof(int));
	    cudaMemcpy(f, f_host, M * N * X * sizeof(int), cudaMemcpyHostToDevice);
	    cudaMalloc((void**)&f_out, M * N * X * sizeof(int));
	    
    	validar(f_host, N, M);

	    cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);

	    // Iteraciones de time step
	    for (int j=0; j<1; j++){
	    	if (i == 0){
	    		kernelAoS_col<<<gs, bs>>>(f, f_out, X, N, M);
	    		kernelAoS_stream<<<gs, bs>>>(f, f_out, j, N, M);
	    	}
	    	else{
	    		kernelSoA_col<<<gs, bs>>>(f, f_out, X, N, M);
	    		// kernelSoA_stream<<<gs, bs>>>(f, f_out, X, N, M);
	    	}
	    }

	    cudaEventRecord(ct2);
	    cudaEventSynchronize(ct2);
	    cudaEventElapsedTime(&dt, ct1, ct2);
	    std::cout << "Tiempo GPU: " << dt << "[ms]" << std::endl;
	    f_hostout = new int[M * N * X];
	    // cudaMemcpy(f_hostout, f, M * N * X * sizeof(int), cudaMemcpyDeviceToHost);
	    cudaMemcpy(f_hostout, f, M * N * X * sizeof(int), cudaMemcpyDeviceToHost);

	    for (int j=0; j<1; j++){
	    	if (i == 0){
	    		Write_AoS(f_hostout, M, N, "initial_f.txt\0");
	    	}
	    	else{
	    		Write_SoA(f_hostout, M, N, "initial_f.txt\0");
	    	}
	    }

	    validar(f_hostout, N, M);

    	cudaFree(f);
    	cudaFree(f_out);
    	delete[] f_host;
    	delete[] f_hostout;
	}
	return 0;
}