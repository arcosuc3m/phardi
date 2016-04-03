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

#ifndef INTRAVOX_H
#define INTRAVOX_H

#include <arrayfire.h>
#include <plog/Log.h>
#include <armadillo>
#include <iostream>
#include <cmath>

namespace phardi {
  
    template<typename T>
    af::array mBessel_ratio(T n, const af::array & x) {
        using namespace arma;
	using namespace af;

        af::array y(x.dims(0),x.dims(1));

        // y = x./( (2*n + x) - ( 2*x.*(n+1/2)./ ( 2*n + 1 + 2*x - ( 2*x.*(n+3/2)./ ( 2*n + 2 + 2*x - ( 2*x.*(n+5/2)./ ( 2*n + 3 + 2*x ) ) ) ) ) ) );
        y = x/( (2*n + x) - ( 2*x*(n+1.0/2.0)/ ( 2.0*n + 1.0 + 2.0*x - ( 2.0*x*(n+3.0/2.0) / ( 2.0*n + 2.0 + 2.0*x - ( 2.0*x*(n+5.0/2.0) / ( 2.0*n + 3.0 + 2.0*x ) ) ) ) ) ) );

        return y;
    }

    template<typename T>
    arma::Mat<T> intravox_fiber_reconst_sphdeconv_rumba_sd(const arma::Mat<T> & Signal,
                                                           const arma::Mat<T> & Kernel,
                                                           const arma::Mat<T> & fODF0,
                                                           const int Niter,
							   T & mean_SNR) {
        using namespace arma;
	using namespace af;

        Mat<T> fODF;

        // n_order = 1;
        T n_order = 1;

        // if Dim(2) > 1
        if (Signal.n_cols > 1) {
            //  fODF = repmat(fODF0, [1, Dim(2)]);
            fODF = repmat(fODF0, 1, Signal.n_cols);
        }
        else {
            //  fODF = fODF0;
            fODF = fODF0;
        }

        // sigma0 = 1/15;
        T sigma0 = 1.0/15.0;

	size_t cols = Signal.n_cols;

        //N = Dim(1);
        int N = Signal.n_rows;

        af::array fODFi_cuda;
        af::array Ratio_cuda;
        af::array RL_factor_cuda;

        // sigma2 = sigma0^2;
	af::array sigma2_cuda = af::constant(std::pow(sigma0,2), dim4(Signal.n_rows, Signal.n_cols)); 
	af::array sigma2_i_cuda;

	af::array fODF_cuda = af::array(fODF.n_rows,fODF.n_cols,fODF.memptr());
	af::array Signal_cuda = af::array(Signal.n_rows,Signal.n_cols,Signal.memptr());
	af::array Kernel_cuda = af::array(Kernel.n_rows,Kernel.n_cols,Kernel.memptr());

	af::array KernelT_cuda = transpose (Kernel_cuda);    

	//Reblurred = Kernel*fODF;
	af::array Reblurred_cuda = matmul(Kernel_cuda,fODF_cuda);

	af::array Reblurred_S_cuda = Signal_cuda * Reblurred_cuda / sigma2_cuda;


        // for i = 1:Niter
        for (size_t i = 0; i < Niter; ++i) {
            // fODFi = fODF;
            fODFi_cuda = fODF_cuda;

            //Ratio = mBessel_ratio(n_order,Reblurred_S);
            Ratio_cuda = mBessel_ratio<T>(n_order,Reblurred_S_cuda);

	    // RL_factor = KernelT*( Signal.*( Ratio ) )./( KernelT*(Reblurred) + eps);
            RL_factor_cuda = matmul(KernelT_cuda, (Signal_cuda * Ratio_cuda)) / (matmul(KernelT_cuda, Reblurred_cuda) + std::numeric_limits<double>::epsilon());

            //fODF = fODFi.*RL_factor;
            fODF_cuda = fODFi_cuda * RL_factor_cuda;

            //% --------------------- Update of variables ------------------------- %
            //Reblurred = Kernel*fODF;
            Reblurred_cuda = matmul(Kernel_cuda, fODF_cuda);

            //Reblurred_S = (Signal.*Reblurred)./sigma2;
            Reblurred_S_cuda = (Signal_cuda * Reblurred_cuda) / sigma2_cuda;

            //% -------------------- Estimate the noise level  -------------------- %
            //sigma2_i = (1/N)*sum( (Signal.^2 + Reblurred.^2)/2 - (sigma2.*Reblurred_S).*Ratio, 1)./n_order;
	    sigma2_i_cuda = (1.0/N) * af::sum( (af::pow(Signal_cuda,2) + af::pow(Reblurred_cuda,2))/2 - (sigma2_cuda * Reblurred_S_cuda) * Ratio_cuda , 0) / n_order;
   
            //sigma2_i = min((1/10)^2, max(sigma2_i,(1/50)^2)); % robust estimator on the interval sigma = [1/SNR_min, 1/SNR_max],
	    for (size_t kk = 0; kk < cols; kk += cols/2) {
 		 gfor (array j, kk, kk+(cols/2) -1)
	    	     sigma2_i_cuda(j) =  af::min(std::pow<T>(1.0/10.0,2), af::max(sigma2_i_cuda(j), std::pow<T>(1.0/50.0,2)));
	    }

            //% where SNR_min = 10 and SNR_max = 50
	    gfor(seq j, N)
		sigma2_cuda(j,af::span) = sigma2_i_cuda(af::span);
		

        }

	//mean_SNR = mean( 1./sqrt( sigma2_i ) );  
	af::array SNR = af::mean (1.0 / af::sqrt(sigma2_i_cuda));
	mean_SNR = SNR(0).scalar<T>();

	fODF_cuda.host(fODF.memptr());
	//Normalization
	//fODF = fODF./repmat( sum(fODF,1) + eps, [size(fODF,1), 1] ); %  
	fODF = fODF / repmat(sum(fODF,0) + std::numeric_limits<double>::epsilon(), fODF.n_rows  ,1);

        return fODF;
    }

    
}

#endif
