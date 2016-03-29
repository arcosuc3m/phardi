/*
Copyright (c) 2016 
Javier Garcia Blas (fjblas@inf.uc3m.es)
Jose Daniel Garcia Sanchez (josedaniel.garcia@uc3m.es)
Yasser Aleman (yaleman@hggm.es)
Erick Canales (ejcanalesr@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the 
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN 
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/


#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <thrust/copy.h>
#include <thrust/device_vector.h>

namespace phardi {
   

    unsigned udivup(unsigned a, unsigned b)
    {
        return (a % b) ? a / b + 1 : a / b;
    }

    template<typename T>
    void gpu_blas_mmul(cublasHandle_t &handle, const T *A, const T *B, T *C, const int m, const int k, const int n) {
        int lda=m,ldb=k,ldc=m;
        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;

        // Do the actual multiplication
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    template<typename T>
    void gpu_blas_mtrans(cublasHandle_t &handle, const T *A, T *B, const int m,  const int n) {
        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;

	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, alpha, B, n, beta, clone, m, A, m );
    }

    template<typename T>
    void gpu_vmul(const T *A, const T *B, T *C, const size_t m) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= m) return;

	C[idx] = A[idx] * B[idx];
    }

    template<typename T>
    void gpu_vdiv(const T *A, const T n, T *C, const size_t m) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= m) return;

        C[idx] = A[idx] / n;
    }


    template<typename T>
    __global__ void mBessel_ratio_kernel(T n, const T * x, T * y, const size_t m) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= m) return;

        y[idx] = x[idx] / ((2*n + x[idx]) - (2*x[idx]*(n+1.0/2.0) / (2*n + 1 +2*x[idx] - (2*x[idx]*(n+3.0/2.0) / (2*n + 2 + 2*x[idx] - (2*x[idx]*(n+5.0/2.0) / (2*n +3 +2 *x[idx])))))));
    }


    template<typename T>
    __global__ void noise_level_kernel(const T *Signal, const T *Reblurred, const T *Reblurred_S, const T *Ratio, T sigma2, T *Reduce, const size_t m) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= m) return;

        Reduce[idx] = std::pow<T>(Signal[idx],2) + std::pow<T>(Reblurred[idx],2)/2 - (sigma2 * Reblurred_S[idx]) * Ratio[idx];

    }

    template<typename T>
    __global__ void Reblurred_S_kernel(const T *Signal, const T *Reblurred, const T sigma, T* Reblurred_S, const size_t m) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= m) return;

        Reblurred_S[idx] = Signal[idx] * Reblurred[idx] / sigma;

    }

    template<typename T>
    __global__ void RL_factor_kernel(const T *Signal, const T *Reblurred,  const T sigma, T *fODF, const size_t m) {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx >= m) return;


    }

    template<typename T>
    void intravox_fiber_reconst_sphdeconv_rumba_sd_kernel (const std::vector<T> & Signal, 
							   const std::vector<T> & Reblurred, 
							   const std::vector<T> & Kernel,
                                     			   std::vector<T> & fODF, 
 							   const size_t Signal_size,
							   const int Niter) { 
        cublasHandle_t handle;
        cublasCreate(&handle);

        T n_order = 1;

    	const int BLOCKSIZE_1D = 256; 

	dim3 block( BLOCKSIZE_1D, 1);
        dim3 grid (udivup(Signal_size, block.x), 1);

        T sigma0 = 1.0/15.0;
        T sigma2 = std::pow(sigma0,2);

        int N = 2;

        thrust::device_vector<T> RL_factor();

        //Mat<T> RL_factor(KernelT.n_rows,Reblurred.n_cols);

        thrust::device_vector<int> d_fODF(fODF.begin(), fODF.end());
        thrust::device_vector<int> d_Signal(Signal.begin(), Signal.end());
        thrust::device_vector<int> d_Kernel(Kernel.begin(), Kernel.end());
        thrust::device_vector<int> d_Reblurred(Reblurred.begin(), Reblurred.end());

        thrust::device_vector<int> d_Reblurred_S(Signal_size);
	thrust::device_vector<int> d_fODFi(d_fODF.size());
	thrust::device_vector<int> d_RL_factor();
	thrust::device_vector<int> d_Ratio(Signal_size);
	thrust::device_vector<int> d_Reduce(Signal_size);

        thrust::device_vector<int> d_KernelT(d_Kernel.size());

	gpu_blas_mtrans<T>(handle, thrust::raw_pointer_cast(d_Kernel.data()),
				   thrust::raw_pointer_cast(d_KernelT.data()), 
					const int m,  const int n);

	Reblurred_S_kernel<T><<< grid, block >>>(thrust::raw_pointer_cast(d_Signal.data()),
                                    thrust::raw_pointer_cast(d_Reblurred.data()),
                                    sigma2,  
                                    thrust::raw_pointer_cast(d_Reblurred_S.data()),
                                    Signal_size); 

	

        for (size_t i = 0; i < Niter; ++i) {
            T sigma2_i;

            d_fODFi = d_fODF;

            mBessel_ratio_kernel<T><<< grid, block  >>>(n_order, 
							thrust::raw_pointer_cast(d_Reblurred_S.data()), 
				 			thrust::raw_pointer_cast(d_Ratio.data()),
							Signal_size);

            //RL_factor = KernelT * (Signal % Ratio) / ((KernelT * Reblurred) + std::numeric_limits<double>::epsilon());

            //fODF = fODFi % RL_factor;
            gpu_vmul<T><<< grid, block >>>(thrust::raw_pointer_cast(d_fODFi.data()),
			    		   thrust::raw_pointer_cast(d_RL_factor.data()),
			    	  	   thrust::raw_pointer_cast(d_fODF.data()),
			    		   Signal_size);

            gpu_blas_mmul<T>(handle, thrust::raw_pointer_cast(&d_Kernel[0]), 
				     thrust::raw_pointer_cast(&d_fODF[0]), 
				     thrust::raw_pointer_cast(&d_Reblurred[0]), );

	    Reblurred_S_kernel<T><<< grid, block >>>(thrust::raw_pointer_cast(d_Signal.data()), 
                                    		     thrust::raw_pointer_cast(d_Reblurred.data()), 
                                    		     sigma2,  
                                    		     thrust::raw_pointer_cast(d_Reblurred_S.data()), 
                                    		     Signal_size); 

	    noise_level_kernel<T><<< grid, block >>>(thrust::raw_pointer_cast(d_Signal.data()),
						     thrust::raw_pointer_cast(d_Reblurred_S.data()),
						     thrust::raw_pointer_cast(d_Ratio.data()),
						     sigma2,
						     thrust::raw_pointer_cast(d_Reduce.data()), 
					             Signal_size);

            T sum;
	    sum = thrust::reduce(d_Reduce.begin(), d_Reduce.end());

            sigma2_i =(1.0/N) * sum / n_order;
            sigma2 = std::min<T>(std::pow<T>(1.0/10.0,2),std::max<T>(sigma2_i, std::pow<T>(1.0/50.0,2)));
        }

	thrust::copy(d_fODF.begin(), d_fODF.end(), fODF.begin());

        cublasDestroy(handle);
     }
}
