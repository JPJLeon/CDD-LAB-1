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

    if (tipo != 0){ // AoS
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
	    	for(int i = 0; i < Largo; i++){
		        fscanf(fp, "%d ", &(f1[i + j*Largo]));
		        // printf("%d ", f1[i + j*Largo]);
	    	}
	    	// printf("\n");
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
    //printf("\n");
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
	        fprintf(fp, "%d ", f[i + j*Largo]);
	    	// printf("%d ", f[i + j*Largo]);
	    }
	    fprintf(fp, "%d\n", f[Largo-1 + j*Largo]);
	    // printf("%d\n", f[Largo-1 + j*Largo]);
    }
    //printf("\n");
    fclose(fp);
}

void validar(int *f, int N, int M, int i){
	int suma=0;
	for(int i=0; i<N*M*4; i++){
		suma += f[i];
  }
  if (i == 0){
    printf("Cantidad inicial de particulas: %d\n", suma);
  } else if (i == 1){
    printf("Cantidad final de particulas: %d\n", suma);
    printf("\n");
  }
	
}

//funcion auxiliar %, funciona con entradas negativas
__device__ int modulo(int a, int b){
    //a%b
    if (a >= 0){
        return a %b;
    }
    return b + a;
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
__global__ void kernelAoS_stream(int *f, int *f_out, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N*M){
		// f0: der
		// f1: arr
		// f2: izq
		// f3: abj
		int x, y, idb;
		idb = tid*4;        //indice del f0 en el arreglo
		x = tid % M; // 4
		y = tid / M; // 1
		// Id de los nodos adyacentes
		int nd[] = {modulo(x+1,M)  + y              *M, 
					x              + modulo(y+1, N) *M, 
					modulo(x-1, M) + y              *M, 
					x              + modulo(y-1, N) *M };
		// Recorremos las direcciones
		for(int i=0; i<4; i++){
			// Seteo todas en 0
			//f_out[idb+i] = 0;
			// Si la particula se mueve en esta direccion
			if(f[idb+i] == 1){
				// La direccion del nodo de esa direccion cambia
				f_out[nd[i]*4+i] = 1;
			}
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
		int x, y, Largo = N*M;
		x = tid % M; // 4
		y = tid / M; // 1
		// Id de los nodos adyacentes
		int nd[] = { modulo(x+1,M) + y*M,
					x + modulo(y+1,N)*M,
					modulo(x-1,M) + y*M,
					x + modulo(y-1,N)*M };
		// Recorremos las direcciones
		for(int i=0; i<X; i++){
			// Seteo todas en 0
			//f_out[tid + i*Largo] = 0;
			// Si la particula se mueve en esta direccion
			if(f[tid + i*Largo] == 1){
				// La direccion del nodo de esa direccion cambia
				f_out[nd[i] + i*Largo] = 1;
			}
		}
	}
}

__global__ void f_out_0(int *f_out, int N, int M){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N*M*4){
    f_out[tid] = 0;
	}
}

//--------------------------------------------------------------------------------

//Pregunta 2, condiciones de borde con AoS

/*  Procesamiento GPU AoS Coalisiones */
__global__ void kernelAoS_col_borde(int *f, int *f_out, int X, int N, int M, int j){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < M*N){
		int idb = tid*4;
		int f0, f1, f2, f3, x, y;
	    x = tid % M; 
		y = tid / M; 

	    // Almacenamos los datos en memoria
	    f0 = f[idb+0];
	    f1 = f[idb+1];
	    f2 = f[idb+2];
	    f3 = f[idb+3];
 
	    bool borde =  (x == 0 || x == M -1 || y == 0 || y == N-1) ;
	    bool horizontal = f0 && f2 && f1 == 0 && f3 == 0;
	    bool vertical = f0 == 0 && f2 == 0 && f1 && f3;

	    //if statement
	    if (j == 0){
			if ( !borde ){ //si es que no se está en algun borde
				if(horizontal){
					f[idb] = 0;
					f[idb+1] = 1;
					f[idb+2] = 0;
					f[idb+3] = 1;
				} else if(vertical){
					f[idb] = 1;
					f[idb+1] = 0;
					f[idb+2] = 1;
					f[idb+3] = 0;
				}
	      	}
	    }

	    //operador ternario
	    else if (j == 1){
	        
	        f[idb] = (borde ? f[idb] : 
	                  (horizontal ? 0:
	                   (vertical ? 1 : f[idb])));
	        f[idb+1] = (borde ? f[idb +1 ] : 
	                  (horizontal ? 1:
	                   (vertical ? 0 : f[idb+ 1])));
	        f[idb+2] = (borde ? f[idb +2] : 
	                  (horizontal ? 0:
	                   (vertical ? 1 : f[idb +2])));
	        f[idb+3] = (borde ? f[idb +3] : 
	                  (horizontal ? 1:
	                   (vertical ? 0 : f[idb + 3])));
	    }

	    //operador booleano
	    else if (j == 2){
	        f[idb] 	  =  (borde) * f[idb]  	  +  abs(borde -1)  * ((horizontal) * 0 + abs(horizontal-1) * ((vertical) * 1 + abs(vertical -1) * f[idb]));
	        f[idb+ 1] =  (borde) * f[idb+ 1]  +  abs(borde -1)  * ((horizontal) * 1 + abs(horizontal-1) * ((vertical) * 0 + abs(vertical -1) * f[idb+ 1]));
	        f[idb+ 2] =  (borde) * f[idb+ 2]  +  abs(borde -1)  * ((horizontal) * 0 + abs(horizontal-1) * ((vertical) * 1 + abs(vertical -1) * f[idb+ 2]));
	        f[idb+ 3] =  (borde) * f[idb+ 3]  +  abs(borde -1)  * ((horizontal) * 1 + abs(horizontal-1) * ((vertical) * 0 + abs(vertical -1) * f[idb+ 3]));
	    }
	}
}

/*  Procesamiento GPU AoS Streaming */
__global__ void kernelAoS_stream_borde(int *f, int *f_out, int N, int M, int j){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	if(tid < N*M){
		int x, y, idb;
		idb = tid*4;        //indice del f0 en el arreglo
		x = tid % M; 
		y = tid / M; 
		// Id de los nodos adyacentes
		int nd[] = {modulo(x+1,M)  + y*M, 
					x + modulo(y+1, N) *M, 
					modulo(x-1, M) + y*M, 
					x + modulo(y-1, N) *M };

		// if statement
	    if (j == 0){
			for(int i=0; i<4; i++){
				// f0: der
				// f1: arr
				// f2: izq
				// f3: abj

				//condiciones de borde
				bool der = (x == M-1 && i == 0);
				bool arr = (y == N-1 && i == 1);
				bool izq = (x == 0 && i == 2);
				bool abj = (y == 0 && i==3);
				// Si la particula se mueve en esta direccion
				if(f[idb+i] == 1){                               //si fi == 0
				    if (abj){                         			 //si se mueve hacia abajo en el borde inferior
				        f_out[nd[1] * 4 + 1] = 1;                //rebota hacia arriba 
				    }
				    else if (arr){
				        f_out[nd[3] * 4 + 3] = 1;
				    }
				    else if (der){
				        f_out[nd[0] *4 + 0] = 1;
				    } 
				    else if (izq){
				        f_out[nd[2] * 4 + 2] = 1;
				    }
				    else{
				        f_out[nd[i]*4+i] = 1;      
				    }
				}
			}
	    }

	    //operador ternario
	    else if(j == 1){
			for(int i=0; i<4; i++){
				// Seteo todas en 0
				//f_out[idb+i] = 0;

				bool der = (x == M-1 && i == 0);
				bool arr = (y == N-1 && i == 1);
				bool izq = (x == 0 && i == 2);
				bool abj = (y == 0 && i==3);

				!(f[idb+i] == 1) ? true : 
				    (abj) ?  f_out[nd[1] * 4 + 1] = 1 :
				      	(arr) ? f_out[nd[3] * 4 + 3] = 1:
				        	(der) ? f_out[nd[0] *4 + 0] = 1 : 
				          		(izq) ? f_out[nd[2] * 4 + 2] = 1 : f_out[nd[i]*4+i] = 1;
			}
		}

		//operador booleano
		else if (j == 2){
			for(int i=0; i<4; i++){
				// Seteo todas en 0
				//f_out[idb+i] = 0;

				bool activo = (f[idb+i] == 1);
				bool der = (x == M-1 && i == 0);
				bool arr = (y == N-1 && i == 1);
				bool izq = (x == 0 && i == 2);
				bool abj = (y == 0 && i==3);

				f_out[nd[1] * 4 + 1]  = activo * abj; 
				f_out[nd[3] * 4 + 3] = activo * abs(abj-1) * arr;
				f_out[nd[0] *4 + 0]   = activo * abs(abj-1) * abs(arr-1) * der; 
				f_out[nd[2] * 4 + 2]  = activo * abs(abj-1) * abs(arr-1) * abs(der - 1) * izq;
				f_out[nd[i]*4+i]      = activo * abs(abj-1) * abs(arr-1) * abs(der - 1) * abs(izq-1);
			}
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
	const char *metodo;
	int M, N;
    int *f_host, *f_hostout, *f, *f_out, *temp;
    char filename[15] = "initial.txt\0";
	int gs, bs = 256;
	int X = 4;

  	// Ejecucion pregunta 1
	// 2 metodos SoA y AoS
    for (int i=0; i<2; i++){
    	if(i==0){
    		continue;
    	}
    	Read(&f_host, &M, &N, filename, X, i);

	    gs = (int)ceil((float) M * N * X / bs);    
	    cudaMalloc((void**)&f, M * N * X * sizeof(int));
	    cudaMemcpy(f, f_host, M * N * X * sizeof(int), cudaMemcpyHostToDevice);
	    cudaMalloc((void**)&f_out, M * N * X * sizeof(int));
	    cudaMalloc((void**)&temp, M * N * X * sizeof(int));
    	validar(f_host, N, M);

	    cudaEventCreate(&ct1);
	    cudaEventCreate(&ct2);
	    cudaEventRecord(ct1);

	    // Iteraciones de time step 
	    for (int j=0; j<1; j++){
        	f_out_0<<<gs, bs>>>(f_out, N, M);
	    	if (i == 0){
	    		kernelSoA_col<<<gs, bs>>>(f, f_out, X, N, M);
	    		kernelSoA_stream<<<gs, bs>>>(f, f_out, X, N, M);
	    	}
	    	else{
	    		kernelAoS_col<<<gs, bs>>>(f, f_out, X, N, M);
	    		kernelAoS_stream<<<gs, bs>>>(f, f_out, N, M);
	    	}
	    	//memory swap
			temp = f;
			f = f_out;
			f_out = temp;
	    }
      
		cudaEventRecord(ct2);
		cudaEventSynchronize(ct2);
		cudaEventElapsedTime(&dt, ct1, ct2);
		f_hostout = new int[M * N * X];
		cudaMemcpy(f_hostout, f, M * N * X * sizeof(int), cudaMemcpyDeviceToHost);

		if (i == 0){
			Write_SoA(f_hostout, M, N, "initial_S.txt\0");
			metodo = "SoA";
		}
		else{
			Write_AoS(f_hostout, M, N, "initial_A.txt\0");
			metodo = "AoS";
		}

		std::cout << "Tiempo " << metodo << ": " << dt << "[ms]" << std::endl;
		validar(f_hostout, N, M, 1);

	    cudaFree(f);
	    cudaFree(temp);
	    cudaFree(f_out);
	    delete[] f_host;
	    delete[] f_hostout;
	}
  
	// Ejecucion pregunta 2
	// metodo AoS con if, terciario y booleano
	// Matriz con bordes
	for (int i=0; i<3; i++){
		// if(i != 2){
		// 	continue;
		// }
		Read(&f_host, &M, &N, filename, X, 0);

		gs = (int)ceil((float) M * N * X / bs);    
		cudaMalloc((void**)&f, M * N * X * sizeof(int));
		cudaMemcpy(f, f_host, M * N * X * sizeof(int), cudaMemcpyHostToDevice);
		cudaMalloc((void**)&f_out, M * N * X * sizeof(int));
		cudaMalloc((void**)&temp, M * N * X * sizeof(int));
		validar(f_host, N, M, 0);

		cudaEventCreate(&ct1);
		cudaEventCreate(&ct2);
		cudaEventRecord(ct1);

		// Iteraciones de time step 
		for (int j=0; j<1; j++){
        	f_out_0<<<gs, bs>>>(f_out, N, M);
	    	kernelAoS_col_borde<<<gs, bs>>>(f, f_out, X, N, M, i);
			kernelAoS_stream_borde<<<gs, bs>>>(f, f_out, N, M, i);
	    	//memory swap
			temp = f;
			f = f_out;
			f_out = temp;
	    }

		//memory swap
		temp = f;
		f = f_out;
		f_out = temp;
		cudaDeviceSynchronize();
		 
		cudaEventRecord(ct2);
		cudaEventSynchronize(ct2);
		cudaEventElapsedTime(&dt, ct1, ct2);
		f_hostout = new int[M * N * X];
		cudaMemcpy(f_hostout, f, M * N * X * sizeof(int), cudaMemcpyDeviceToHost);

	    if (i == 0){
	    	metodo = "IF ";
	    }
	    else if (i == 1){
	    	metodo = "TERNARIO ";
	    } 
	    else if (i == 2) {
			metodo = "BOOLEANO ";
	    }

		std::cout << "Tiempo AoS con bordes y operador: " << metodo << dt << "[ms]" << std::endl;
		validar(f_hostout, N, M, 1);

		cudaFree(f);
		cudaFree(temp);
		cudaFree(f_out);
		delete[] f_host;
		delete[] f_hostout;
	}

  	// Ejecucion pregunta 3

	return 0;
}