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

#ifndef CREATE_KERNEL_RUMBA_H
#define CREATE_KERNEL_RUMBA_H


#include "create_signal_multi_tensor.hpp"
#include "common.hpp"

#include <plog/Log.h>
#include <iostream>
#include <math.h>
#include <armadillo>

namespace phardi {


    template <typename T>
    void create_Kernel_for_rumba(const arma::Mat<T> & V,
                                 const arma::Mat<T> & diffGrads,
                                 const arma::Col<T> & diffBvals,
                                 arma::Mat<T> & Kernel,
                                 const phardi::options opts) {


        using namespace arma;
        // add_rician_noise = 0;
        bool add_rician_noise = opts.rumba_sd.add_noise;

        T SNR = 1;

        // [phi, theta] = cart2sph(V(:,1),V(:,2),V(:,3)); % set of directions
        Col<T> phi(V.n_rows);
        Col<T> theta(V.n_rows);

	#pragma omp parallel for
        for (int i = 0; i < V.n_rows; ++i) {
            Cart2Sph(V(i,0),V(i,1),V(i,2),phi(i),theta(i));
        }

	    theta.transform( [](T val) { return (-val); } );

        //S0 = 1; % The Dictionary is created under the assumption S0 = 1;
        T S0 = 1;
        //fi = 1; % volume fraction
        Col<T> fi(1);
        fi(0) = 1;

        Col<T> S(diffGrads.n_rows);
        Mat<T> D(3,3);
        Mat<T> v2(fi.n_elem,2);
        Col<T> v3(3);

        T c = 180/M_PI;

        //for i=1:length(phi)
        for (size_t i = 0; i < phi.n_elem; i++) {
            // anglesFi = [phi(i), theta(i)]*(180/pi); % in degrees
            v2(0,0) = phi(i)*c; v2(0,1) = theta(i)*c;
            v3(0) = opts.rumba_sd.lambda1; v3(1) = opts.rumba_sd.lambda2;  v3(2) = opts.rumba_sd.lambda2;

            S.fill(0.0);
            // Kernel(:,i) = create_signal_multi_tensor(anglesFi, fi, [lambda1, lambda2, lambda2], diffBvals, diffGrads, S0, SNR, add_rician_noise);            
            create_signal_multi_tensor<T>(v2, fi, v3, diffBvals, diffGrads, S0, SNR, add_rician_noise, S, D);
            for (size_t j = 0; j<S.n_elem;++j) {
                Kernel(i,j) = S(j);
            }
        }

        S.fill(0.0);
        v2(0,0) = phi(0)*c; v2(0,1) = theta(0)*c;
        v3(0) = opts.rumba_sd.lambda_csf; v3(1) = opts.rumba_sd.lambda_csf;  v3(2) = opts.rumba_sd.lambda_csf;
        create_signal_multi_tensor<T>(v2, fi, v3, diffBvals, diffGrads, S0, SNR, add_rician_noise, S, D);

	    #pragma omp parallel for
        for (size_t i = 0; i<S.n_elem; ++i) {
            Kernel(phi.n_elem  ,i) = S(i);
        }

        S.fill(0.0);
        v2(0,0) = phi(0)*c; v2(0,1) = theta(0)*c;
        v3(0) = opts.rumba_sd.lambda_gm; v3(1) = opts.rumba_sd.lambda_gm;  v3(2) = opts.rumba_sd.lambda_gm;
        create_signal_multi_tensor<T>(v2, fi, v3, diffBvals, diffGrads, S0, SNR, add_rician_noise, S, D);

	    #pragma omp parallel for
        for (size_t i = 0; i<S.n_elem; ++i) {
            Kernel(phi.n_elem + 1 ,i) = S(i);
        }

        Kernel = Kernel.t();
    }
}

#endif
