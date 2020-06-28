#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 1000000

//generar arreglo
void GenArray(int **array)
{
    int *array1 = new int[N];
    for (int i = 0; i < N; i++)
        array1[i] = rand() % 1000;
    *array = array1;
}

//procesamiento CPU
void FindMax(int *arr, int *max)
{
    *max = 0;
    for (int i = 0; i < N; i++)
    {
        if (*max < arr[i])
            *max = arr[i];
    }
}

//Procesamiento GPU

//pregunta 2
__global__ void FindMax2(int *arr, int *max)
{
    int tid = threadIdx.x;
    int local_max = 0;
    for (int i = tid * 97657; i < (tid + 1) * 97657 && i < N; i++)
    {
        if (local_max < arr[i])
        {
            local_max = arr[i];
        }
    }

    //una vez todos los threads del bloque terminan de encontrar su máximo, se hace la reducción
    __shared__ int arrSM[1024];
    arrSM[tid] = local_max;

    int reduccion = 1024 / 2;
    while (reduccion > 0)
    {
        __syncthreads();
        if (tid < reduccion && arrSM[tid] < arrSM[tid + reduccion])
            arrSM[tid] = arrSM[tid + reduccion];
        reduccion /= 2;
    }
    if (tid == 0)
        *max = arrSM[0];
}

//pregunta 3
__global__ void FindMax3(int *arr, int *max, int largo_arreglo)
{
    //cada hebra copia un dato en el arreglo de memoria compartida
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < largo_arreglo)
    {
        __shared__ int arrSM[256];
        arrSM[threadIdx.x] = arr[tid];

        //reduccion
        int reduccion = 256 / 2;
        while (reduccion > 0)
        {
            __syncthreads();
            if (threadIdx.x < reduccion && arrSM[threadIdx.x] < arrSM[threadIdx.x + reduccion])
                arrSM[threadIdx.x] = arrSM[threadIdx.x + reduccion];
            reduccion /= 2;
        }

         //si es que solo hay un bloque, se retorna el resultado final
        if (largo_arreglo < 256){
            *max = arrSM[0];
        //si es el primer thread del bloque
        } else if (threadIdx.x == 0) { 
            arr[blockIdx.x] = arrSM[0];
        }
    }
}

//pregunta 4
__global__ void FindMax4(int *arr, int *max)
{
    //cada hebra copia un dato en el arreglo de memoria compartida
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        __shared__ int arrSM[256];
        arrSM[threadIdx.x] = arr[tid];

        //reduccion
        int reduccion = 256 / 2;
        while (reduccion > 0)
        {
            __syncthreads();
            if (threadIdx.x < reduccion && arrSM[threadIdx.x] < arrSM[threadIdx.x + reduccion])
                arrSM[threadIdx.x] = arrSM[threadIdx.x + reduccion];
            reduccion /= 2;
        }

        // __syncthreads();
        if (threadIdx.x == 0)
            atomicMax(max, arrSM[0]);
    }
}

/*-------------------------*/
int main(int argc, char **argv)
{

    clock_t t1, t2;
    double ms;
    cudaEvent_t ct1, ct2;
    float dt;
    int *Ahost;
    int *A;
    int gs = 1, bs = 1024;
    GenArray(&Ahost);

    //CPU
    int max_total;
    t1 = clock();
    
    FindMax(Ahost, &max_total);

    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    printf("Tiempo CPU: %f[ms]\n", ms);
    printf("Maximo: %d\n", max_total);

    /*
	 *  Pregunta 1 - Funcion CPU
	*/

    bs = 1024;
    gs = 1;

    max_total = 0;
    int *max;
    cudaMalloc((void **)&max, sizeof(int));
    cudaMalloc((void **)&A, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * sizeof(int), cudaMemcpyHostToDevice);

    /*
     *  Pregunta 2 - Funcion GPU un  bloque
    */

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    FindMax2<<<gs, bs>>>(A, max);

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaMemcpy(&max_total, max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Tiempo GPU Un Bloque: %f[ms]\n", dt);
    printf("Maximo: %d\n", max_total);

    /*
     *  Pregunta 3 - Funcion GPU multiples bloques
    */

    bs = 256;
    int largo_arreglo = N;
    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    while (largo_arreglo < 256)
    {
        FindMax3<<<(int)ceil((float)largo_arreglo / bs), bs>>>(A, max, largo_arreglo);
        largo_arreglo = (int)ceil((float)largo_arreglo / bs);
    }

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaMemcpy(&max_total, max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Tiempo GPU Multiples Bloques: %f[ms]\n", dt);
    printf("Maximo: %d\n", max_total);

    /*
     *  Pregunta 4 - Funcion GPU multiples bloques con op atom 
    */

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    FindMax4<<<gs, bs>>>(A, max); // GS ??

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaMemcpy(&max_total, max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Tiempo GPU Multiples Bloques Op. Atom: %f[ms]\n", dt);
    printf("Maximo: %d\n", max_total);

    cudaFree(A);
    delete[] Ahost;

    return 0;
}