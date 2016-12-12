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


#include <plog/Log.h>
#include <armadillo>
#include <iostream>
#include <cmath>

namespace phardi {

    template<typename T>
    arma::Mat<T> mBessel_ratio(T n, const arma::Mat<T> & x) {
        using namespace arma;

        Mat<T> y(x.n_rows,x.n_cols);

        // y = x./( (2*n + x) - ( 2*x.*(n+1/2)./ ( 2*n + 1 + 2*x - ( 2*x.*(n+3/2)./ ( 2*n + 2 + 2*x - ( 2*x.*(n+5/2)./ ( 2*n + 3 + 2*x ) ) ) ) ) ) );
#pragma omp parallel for simd
        for (uword j = 0; j < x.n_cols; ++j) {
            for (uword i = 0; i < x.n_rows; ++i) {
                y.at(i,j) = x.at(i,j) / ((2*n + x.at(i,j)) - (2*x.at(i,j)*(n+1.0/2.0) / (2*n + 1 +2*x.at(i,j) - (2*x.at(i,j)*(n+3.0/2.0) / (2*n + 2 + 2*x.at(i,j) - (2*x.at(i,j)*(n+5.0/2.0) / (2*n +3 +2 *x.at(i,j))))))));
            }
        }

        return y;
    }

    template<typename T>
    arma::Mat<T> intravox_fiber_reconst_sphdeconv_rumba_sd(const arma::Mat<T> & Signal,
                                                           const arma::Mat<T> & Kernel,
                                                           const arma::Mat<T> & fODF0,
                                                           const int Niter,
                                                           T & mean_SNR) {
        using namespace arma;

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

        //Reblurred = Kernel*fODF;
        Mat<T> Reblurred;
        Reblurred = Kernel * fODF;

        //KernelT = Kernel'; % only one time
        Mat<T> KernelT = Kernel.t();

        // sigma0 = 1/15;
        T sigma0 = 1.0/15.0;

        // sigma2 = sigma0^2;
        Mat<T> sigma2 (Signal.n_rows, Signal.n_cols);
        sigma2.fill (std::pow(sigma0,2));

        //N = Dim(1);
        int N = Signal.n_rows;

        // Reblurred_S = (Signal.*Reblurred)./sigma2;
        Mat<T> Reblurred_S(Signal.n_rows, Signal.n_cols);
        Reblurred_S = Signal % Reblurred / sigma2;


        Mat<T> fODFi;
        Mat<T> Ratio;
        Mat<T> RL_factor(KernelT.n_rows,Reblurred.n_cols);
        Mat<T> SR(Signal.n_rows,Signal.n_cols);
        Mat<T> SUM(Signal.n_rows,Signal.n_cols);

        Mat<T> KTSR(KernelT.n_rows, SR.n_cols);
        Mat<T> KTRB(KernelT.n_rows, Reblurred.n_cols);

        Row<T> sigma2_i(Signal.n_cols);
        // for i = 1:Niter
        for (uword i = 0; i < Niter; ++i) {
            Ratio = mBessel_ratio<T>(n_order,Reblurred_S);

#pragma omp parallel for simd
            for (uword k = 0; k < SR.n_cols; ++k ) {
                for (size_t j = 0; j < SR.n_rows; ++j) {
                    SR.at(j,k) = Signal.at(j,k) * Ratio.at(j,k);
                }
            }

            KTSR = KernelT * SR;
            KTRB = KernelT * Reblurred;

            // RL_factor = KernelT * SR / ((KernelT * Reblurred) + std::numeric_limits<double>::epsilon());
#pragma omp parallel for 
            for (uword k = 0; k < RL_factor.n_cols; ++k ) {
                for (uword j = 0; j < RL_factor.n_rows; ++j) {
                    RL_factor.at(j,k) = KTSR.at(j,k) / (KTRB.at(j,k) + datum::eps);
                }
            }

#pragma omp parallel for 
            for (uword k = 0; k < fODF.n_cols; ++k ) {
                for (uword j = 0; j < fODF.n_rows; ++j) {
                    fODF.at(j,k) = fODF.at(j,k) * RL_factor.at(j,k);
                }
            }

            Reblurred = Kernel * fODF;

#pragma omp parallel for
            for (uword k = 0; k < Reblurred_S.n_cols; ++k ) {
                for (uword j = 0; j < Reblurred_S.n_rows; ++j) {
                    Reblurred_S.at(j,k) = (Signal.at(j,k) * Reblurred.at(j,k)) / sigma2.at(j,k);
                }
            }

#pragma omp parallel for 
            for (uword k = 0; k < Signal.n_cols; ++k ) {
                for (uword j = 0; j < Signal.n_rows; ++j) {
                    SUM.at(j,k) = (pow(Signal.at(j,k),2) + pow(Reblurred.at(j,k),2))/2 - (sigma2.at(j,k) * Reblurred_S.at(j,k)) * Ratio.at(j,k) ;
                }
            }

            sigma2_i = (1.0/N) * sum( SUM , 0) / n_order;

#pragma omp parallel
            for (uword k = 0; k < sigma2_i.n_elem; ++k ) {
                sigma2_i.at(k) = std::min<T>(std::pow<T>(1.0/10.0,2),std::max<T>(sigma2_i.at(k), std::pow<T>(1.0/50.0,2)));
            }

            sigma2 = repmat(sigma2_i, N, 1);
        }

        //mean_SNR = mean( 1./sqrt( sigma2_i ) );
        mean_SNR = mean (1.0 / sqrt(sigma2_i));

        //Normalization
        //fODF = fODF./repmat( sum(fODF,1) + eps, [size(fODF,1), 1] ); %
        fODF = fODF / repmat(sum(fODF,0) + datum::eps, fODF.n_rows  ,1);

        return fODF;
    }
}
#endif
