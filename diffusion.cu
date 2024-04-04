
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <cmath>
#include <cassert>
#include "get_walltime.h"
using namespace std;

const unsigned int NG = 2;
const unsigned int BLOCK_DIM_X = 256;

__constant__ float c_a, c_b, c_c;

/********************************************************************************
  Error checking function for CUDA
 *******************************************************************************/
// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
//https://github.com/parallel-forall/code-samples/blob/master/series/cuda-cpp/finite-difference/finite-difference.cu
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

/********************************************************************************
  Do one diffusion step, on the host in host memory
 *******************************************************************************/
void host_diffusion(float* u, float *u_new, const unsigned int n, 
     const float dx, const float dt){

  //First, do the diffusion step on the interior points
  for(int i = NG; i < n-NG;i++){
    u_new[i] = u[i] + dt/(dx*dx) *(
                    - 1./12.f* u[i-2]
                    + 4./3.f * u[i-1]
                    - 5./2.f * u[i]
                    + 4./3.f * u[i+1]
                    - 1./12.f* u[i+2]);
  }

  //Apply the dirichlet boundary conditions
  u_new[0] = -u_new[NG+1];
  u_new[1] = -u_new[NG];

  u_new[n-NG]   = -u_new[n-NG-1];
  u_new[n-NG+1] = -u_new[n-NG-2];
}
/********************************************************************************
  Do one diffusion step, with CUDA
 *******************************************************************************/
__global__ 
void cuda_diffusion(float* u, float *u_new, const unsigned int n, const float dx, const float dt){
    int i = blockDim.x * blockIdx.x + threadIdx.x + NG; // Calculate global thread index, adjusted for ghost cells

    if (i >= NG && i < n - NG) { // Check if the thread corresponds to an interior point
        // Apply the diffusion equation
        u_new[i] = u[i] + dt / (dx * dx) * (
                       - 1.f / 12 * u[i-2]
                       + 4.f / 3  * u[i-1]
                       - 5.f / 2  * u[i]
                       + 4.f / 3  * u[i+1]
                       - 1.f / 12 * u[i+2]);
    }

    // Ensure all threads have written their updates before applying boundary conditions
    __syncthreads();

    // Apply Dirichlet boundary conditions by specific threads to avoid race conditions
    if (i == NG) { // Left boundary
        u_new[0] = -u_new[NG + 1];
        u_new[1] = -u_new[NG];
    }
    else if (i == n - NG - 1) { // Right boundary, adjusted for zero-based indexing
        u_new[n - NG]   = -u_new[n - NG - 1];
        u_new[n - NG + 1] = -u_new[n - NG - 2];
    }
}


/********************************************************************************
  Do one diffusion step, with CUDA, with shared memory
 *******************************************************************************/
__global__ 
void shared_diffusion(float* u, float *u_new, const unsigned int n, const float dx, const float dt){
    // Define the size of the shared memory array. We need to account for the halo regions for the stencil operation.
    extern __shared__ float shared_u[];

    int globalIdx = threadIdx.x + blockIdx.x * blockDim.x; // Global index for the current thread
    int localIdx = threadIdx.x + NG; // Local index within shared memory, adjusted for halo

    // Calculate the indices for loading data into shared memory with halo regions
    int loadIdxStart = blockIdx.x * blockDim.x;
    int loadIdxEnd = loadIdxStart + blockDim.x - 1;

    // Load data into shared memory, including halo regions
    if (globalIdx >= loadIdxStart && globalIdx <= loadIdxEnd) {
        shared_u[localIdx] = u[globalIdx];
        
        // Load halo regions if this thread is responsible
        if(threadIdx.x < NG) {
            // Left halo
            if(blockIdx.x > 0 && (globalIdx - NG) >= 0) {
                shared_u[localIdx - NG] = u[globalIdx - NG];
            }
            // Right halo
            if((globalIdx + NG) < n && (localIdx + NG) < (blockDim.x + 2*NG)) {
                shared_u[localIdx + NG] = u[globalIdx + NG];
            }
        }
    }

    __syncthreads(); // Synchronize to ensure all threads have loaded their parts into shared memory

    // Perform the diffusion step using values from shared memory
    if (globalIdx >= NG && globalIdx < n - NG) { // Ensure we're working within valid range
        u_new[globalIdx] = shared_u[localIdx] + dt / (dx * dx) * (
                            - 1.f / 12 * shared_u[localIdx - 2]
                            + 4.f / 3  * shared_u[localIdx - 1]
                            - 5.f / 2  * shared_u[localIdx]
                            + 4.f / 3  * shared_u[localIdx + 1]
                            - 1.f / 12 * shared_u[localIdx + 2]);
    }

    // Apply the Dirichlet boundary conditions directly in global memory (if this thread corresponds to a boundary)
    if (globalIdx == 0) {
        u_new[0] = -u_new[NG + 1];
        u_new[1] = -u_new[NG];
    } else if (globalIdx == n - NG - 1) {
        u_new[n - NG] = -u_new[n - NG - 1];
        u_new[n - NG + 1] = -u_new[n - NG - 2];
    }
}


/********************************************************************************
  Dump u to a file
 *******************************************************************************/
void outputToFile(string filename, float* u, unsigned int n){

  ofstream file;
  file.open(filename.c_str());
  file.precision(8);
  file << std::scientific;
  for(int i =0; i < n;i++){
    file<<u[i]<<endl;
  }
  file.close();
};

/********************************************************************************
  main
 *******************************************************************************/
int main(int argc, char** argv){

  //Number of steps to iterate
  const unsigned int n_steps = 10;
  //const unsigned int n_steps = 100;
  // const unsigned int n_steps = 1000000;

  //Whether and how ow often to dump data
  const bool outputData = true;
  const unsigned int outputPeriod = n_steps/10;

  //Size of u
  const unsigned int n = (1<<11) +2*NG;
  //const unsigned int n = (1<<15) +2*NG;

  //Block and grid dimensions
  const unsigned int blockDim = BLOCK_DIM_X;
  const unsigned int gridDim = (n-2*NG)/blockDim;

  //Physical dimensions of the domain
  const float L = 2*M_PI;
  const float dx = L/(n-2*NG-1);
  const float dt = 0.25*dx*dx;

  //Create constants for 6th order centered 2nd derivative
  float const_a = 1.f/12.f * dt/(dx*dx);  
  float const_b = 4.f/3.f  * dt/(dx*dx);
  float const_c = 5.f/2.f  * dt/(dx*dx);

  //Copy these to the cuda constant memory
  cudaMemcpyToSymbol(c_a, &const_a, sizeof(float), 0, cudaMemcpyHostToDevice)  ;
  cudaMemcpyToSymbol(c_b, &const_b, sizeof(float), 0, cudaMemcpyHostToDevice) ;
  cudaMemcpyToSymbol(c_c, &const_c, sizeof(float), 0, cudaMemcpyHostToDevice)  ;


  //iterator, for later
  int i;

  //Create cuda timers
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

  //Timing variables
  float milliseconds;
  double startTime,endTime;

  //Filename for writing
  char filename[256];

  //Allocate memory for the initial conditions
  float* initial_u = new float[n];

  //Initialize with a periodic sin wave that starts after the left hand
  //boundaries and ends just before the right hand boundaries
  for( i = NG; i < n-NG; i++){
    initial_u[i] = sin( 2*M_PI/L*(i-NG)*dx);
  }
  //Apply the dirichlet boundary conditions
  initial_u[0] = -initial_u[NG+1];
  initial_u[1] = -initial_u[NG];

  initial_u[n-NG]   = -initial_u[n-NG-1];
  initial_u[n-NG+1] = -initial_u[n-NG-2];

/********************************************************************************
  Test the host kernel for diffusion
 *******************************************************************************/

  //Allocate memory in the host's heap
  float* host_u  = new float[n];
  float* host_u2 = new float[n];//buffer used for u_new

  //Initialize the host memory
  for( i = 0; i < n; i++){
    host_u[i] = initial_u[i];
  }

  outputToFile("data/host_uInit.dat",host_u,n);

  
  get_walltime(&startTime);
  //Perform n_steps of diffusion
  for( i = 0 ; i < n_steps; i++){

    if(outputData && i%outputPeriod == 0){
      sprintf(filename,"data/host_u%08d.dat",i);
      outputToFile(filename,host_u,n);
    }

    host_diffusion(host_u,host_u2,n,dx,dt);

    //Switch the buffer with the original u
    float* tmp = host_u;
    host_u = host_u2;
    host_u2 = tmp;

  }
  get_walltime(&endTime);

  cout<<"Host function took: "<<(endTime-startTime)*1000./n_steps<<"ms per step"<<endl;

  outputToFile("data/host_uFinal.dat",host_u,n);

/********************************************************************************
  Test the cuda kernel for diffusion
 *******************************************************************************/
  //Allocate a copy for the GPU memory in the host's heap
  float* cuda_u  = new float[n];

  //Initialize the cuda memory
  for( i = 0; i < n; i++){
    cuda_u[i] = initial_u[i];
  }
  outputToFile("data/cuda_uInit.dat",cuda_u,n);

  //Allocate memory on the GPU
  float* d_u, *d_u2;
  //FIXME Allocate d_u,d_u2 on the GPU, and copy cuda_u into d_u

	cudaEventRecord(start);//Start timing
  //Perform n_steps of diffusion
  for( i = 0 ; i < n_steps; i++){

    if(outputData && i%outputPeriod == 0){
      //Copy data off the device for writing
      sprintf(filename,"data/cuda_u%08d.dat",i);
      //FIXME
			
      outputToFile(filename,cuda_u,n);
    }

    //Call the cuda_diffusion kernel
    //FIXME

    //Switch the buffer with the original u
    //FIXME

  }
	cudaEventRecord(stop);//End timing
	

  //Copy the memory back for one last data dump
  sprintf(filename,"data/cuda_u%08d.dat",i);
  //FIXME
  
  outputToFile(filename,cuda_u,n);

  //Get the total time used on the GPU
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

  cout<<"Cuda Kernel took: "<<milliseconds/n_steps<<"ms per step"<<endl;


/********************************************************************************
  Test the cuda kernel for diffusion with shared memory
 *******************************************************************************/

  //Allocate a copy for the GPU memory in the host's heap
  float* shared_u  = new float[n];

  /*
  //Initialize the cuda memory
  for( i = 0; i < n; i++){
    shared_u[i] = initial_u[i];
  }
  outputToFile("data/shared_uInit.dat",shared_u,n);

  //Copy the initial memory onto the GPU
  //FIXME copy shared_u to d_u
	


	cudaEventRecord(start);//Start timing
  //Perform n_steps of diffusion
  for( i = 0 ; i < n_steps; i++){

    if(outputData && i%outputPeriod == 0){
      //Copy data off the device for writing
      sprintf(filename,"data/shared_u%08d.dat",i);
      //FIXME
			
      outputToFile(filename,shared_u,n);
    }

    //Call the shared_diffusion kernel
    //FIXME

    //Switch the buffer with the original u
    //FIXME

  }
	cudaEventRecord(stop);//End timing
	

  //Copy the memory back for one last data dump
  sprintf(filename,"data/shared_u%08d.dat",i);
  //FIXME
  

  //Get the total time used on the GPU
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

  cout<<"Shared Memory Kernel took: "<<milliseconds/n_steps<<"ms per step"<<endl;
  */

/********************************************************************************
  Test the cuda kernel for diffusion, with excessive memcpys
 *******************************************************************************/

  /*
  //Initialize the cuda memory
  for( i = 0; i < n; i++){
    shared_u[i] = initial_u[i];
  }

	cudaEventRecord(start);//Start timing
  //Perform n_steps of diffusion
  for( i = 0 ; i < n_steps; i++){

    //Copy the data from host to device
    //FIXME copy shared_u to d_u

    //Call the shared_diffusion kernel
    //FIXME

    //Copy the data from host to device
    //FIXME copy d_u2 to cuda_u


  }
	cudaEventRecord(stop);//End timing
	


  //Get the total time used on the GPU
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

  cout<<"Excessive cudaMemcpy took: "<<milliseconds/n_steps<<"ms per step"<<endl;
  */

  //Clean up the data
  delete[] initial_u;
  delete[] host_u;
  delete[] host_u2;

  delete[] cuda_u;
  delete[] shared_u;

  //FIXME free d_u and d_2
}
