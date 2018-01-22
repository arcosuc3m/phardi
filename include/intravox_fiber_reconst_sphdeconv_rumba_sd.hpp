/**
* @version              pHARDI v0.3
* @copyright            Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license              GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef INTRAVOX_H
#define INTRAVOX_H

#include <arrayfire.h>
#include <plog/Log.h>
#include <armadillo>
#include <iostream>
#include <cmath>

namespace phardi {
  
    template<class ty> af::dtype get_dtype();
    template<> 
    af::dtype get_dtype<float>() { return f32; }
    template<> 
    af::dtype get_dtype<double>() { return f64; }

    template<typename T>
    af::array mBessel_ratio(T n, const af::array & x) {
        using namespace arma;
	    using namespace af;

        af::array y(x.dims(0),x.dims(1));
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
        af::array fODFi_af;
        af::array Ratio_af;
        af::array RL_factor_af;
        // sigma2 = sigma0^2;
	    af::array sigma2_af = af::constant(std::pow(sigma0,2), dim4(Signal.n_rows, Signal.n_cols),get_dtype<T>());
	    af::array sigma2_i_af;
	    af::array fODF_af = af::array(fODF.n_rows,fODF.n_cols,fODF.memptr());
	    af::array Signal_af = af::array(Signal.n_rows,Signal.n_cols,Signal.memptr());
	    af::array Kernel_af = af::array(Kernel.n_rows,Kernel.n_cols,Kernel.memptr());
	    af::array KernelT_af = transpose (Kernel_af);
	    //Reblurred = Kernel*fODF;
	    af::array Reblurred_af = matmul(Kernel_af,fODF_af);
	    af::array Reblurred_S_af = Signal_af * Reblurred_af / sigma2_af;

        // for i = 1:Niter
        for (size_t i = 0; i < Niter; ++i) {
            Ratio_af = mBessel_ratio<T>(n_order,Reblurred_S_af);
            fODF_af = fODF_af * matmul(KernelT_af, (Signal_af * Ratio_af)) / (matmul(KernelT_af, Reblurred_af) + std::numeric_limits<double>::epsilon());
            Reblurred_af = matmul(Kernel_af, fODF_af);
            Reblurred_S_af = (Signal_af * Reblurred_af) / sigma2_af;
	        sigma2_i_af = (1.0/N) * af::sum( (af::pow(Signal_af,2) + af::pow(Reblurred_af,2))/2 - (sigma2_af * Reblurred_S_af) * Ratio_af , 0) / n_order;
	        sigma2_i_af = af::min(std::pow<T>(1.0/10.0,2), af::max(sigma2_i_af, std::pow<T>(1.0/50.0,2)));
	        gfor(seq j, N)
		        sigma2_af(j,af::span) = sigma2_i_af(af::span);
        }

	    //mean_SNR = mean( 1./sqrt( sigma2_i ) );
        Mat<T> sigma2_i( sigma2_i_af.dims(0),  sigma2_i_af.dims(1));
        sigma2_i_af.host(sigma2_i.memptr());
        Mat<T> SNR = arma::mean (1.0 / arma::sqrt(sigma2_i));
	    mean_SNR =  SNR(0);
	    fODF_af.host(fODF.memptr());
	    //Normalization
	    //fODF = fODF./repmat( sum(fODF,1) + eps, [size(fODF,1), 1] ); %
	    fODF = fODF / repmat(sum(fODF,0) + std::numeric_limits<double>::epsilon(), fODF.n_rows  ,1);

        return fODF;
    }
}

#endif
