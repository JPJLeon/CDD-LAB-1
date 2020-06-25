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
__global__ void FindMax1(int *arr, int *max)
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

    int trabajando = 1024 / 2;
    while (trabajando > 0)
    {
        __syncthreads();
        if (tid < trabajando && arrSM[tid] < arrSM[tid + trabajando])
            arrSM[tid] = arrSM[tid + trabajando];
        trabajando /= 2;
    }
    if (tid == 0)
        *max = arrSM[0];
}

//pregunta 3
__global__ void FindMax2(int *arr, int *max, int largo_arreglo)
{
    //cada hebra copia un dato en el arreglo de memoria compartida
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < largo_arreglo)
    {
        __shared__ int arrSM[256];
        arrSM[threadIdx.x] = arr[tid];

        //reduccion
        int trabajando = 256 / 2;
        while (trabajando > 0)
        {
            __syncthreads();
            if (threadIdx.x < trabajando && arrSM[threadIdx.x] < arrSM[threadIdx.x + trabajando])
                arrSM[threadIdx.x] = arrSM[threadIdx.x + trabajando];
            trabajando /= 2;
        }

        if (largo_arreglo < 256)                       //si es que solo hay un bloque, se retorna el resultado final
            *max = arrSM[0] else if (threadIdx.x == 0) //si es el primer thread del bloque
                arr[blockIdx.x] = arrSM[0];
    }
}

//pregunta 4
__global__ void FindMax3(int *arr, int *max)
{
    //cada hebra copia un dato en el arreglo de memoria compartida
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N)
    {
        __shared__ int arrSM[256];
        arrSM[threadIdx.x] = arr[tid];

        //reduccion
        int trabajando = 256 / 2;
        while (trabajando > 0)
        {
            __syncthreads();
            if (threadIdx.x < trabajando && arrSM[threadIdx.x] < arrSM[threadIdx.x + trabajando])
                arrSM[threadIdx.x] = arrSM[threadIdx.x + trabajando];
            trabajando /= 2;
        }

        __syncthreads();
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
    int maxCPU;
    t1 = clock();
    FindMax(Ahost, &maxCPU);
    t2 = clock();
    ms = 1000.0 * (double)(t2 - t1) / CLOCKS_PER_SEC;
    printf("Tiempo CPU: %f[ms]\n", ms);
    printf("Maximo: %d\n", maxCPU);

    /*
	 *  Parte GPU
	 */

    maxCPU = 0;
    int *max;
    cudaMalloc((void **)&max, sizeof(int));
    cudaMalloc((void **)&A, N * sizeof(int));
    cudaMemcpy(A, Ahost, N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&ct1);
    cudaEventCreate(&ct2);
    cudaEventRecord(ct1);

    bs = 1024;
    gs = 1;
    //FindMax1<<<gs, bs>>>(A, max);

    bs = 256;
    int largo_arreglo = N;
    while (largo_arreglo < 256)
    {
        //FindMax2<<<(int)ceil((float)largo_arreglo / bs), bs>>>(A, max, largo_arreglo);
        largo_arreglo = (int)ceil((float)largo_arreglo / bs);
    }

    //FindMax3<<<gs, bs>>>(A, max);

    cudaEventRecord(ct2);
    cudaEventSynchronize(ct2);
    cudaEventElapsedTime(&dt, ct1, ct2);
    cudaMemcpy(&maxCPU, max, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Tiempo GPU: %f[ms]\n", dt);
    printf("Maximo: %d\n", maxCPU);

    cudaFree(A);
    delete[] Ahost;

    return 0;
}