#ifndef INTRAVOX_H
#define INTRAVOX_H


#include <plog/Log.h>
#include <armadillo>
#include <iostream>
#include <cmath>

namespace pfiber {
  
    template<typename T>
    arma::Mat<T> mBessel_ratio(T n, const arma::Mat<T> & x) {
        using namespace arma;

        Mat<T> y(x.n_rows,x.n_cols);

        // y = x./( (2*n + x) - ( 2*x.*(n+1/2)./ ( 2*n + 1 + 2*x - ( 2*x.*(n+3/2)./ ( 2*n + 2 + 2*x - ( 2*x.*(n+5/2)./ ( 2*n + 3 + 2*x ) ) ) ) ) ) );
        #pragma omp parallel for
        for (size_t i = 0; i < x.n_rows; ++i) {
            for (size_t j = 0; j < x.n_cols; ++j) {
                y(i,j) = x(i,j) / ((2*n + x(i,j)) - (2*x(i,j)*(n+1.0/2.0) / (2*n + 1 +2*x(i,j) - (2*x(i,j)*(n+3.0/2.0) / (2*n + 2 + 2*x(i,j) - (2*x(i,j)*(n+5.0/2.0) / (2*n +3 +2 *x(i,j))))))));
            }
        }

        return y;
    }

    template<typename T>
    arma::Mat<T> intravox_fiber_reconst_sphdeconv_rumba_sd(const arma::Mat<T> & Signal,
                                                           const arma::Mat<T> & Kernel,
                                                           const arma::Mat<T> & fODF0,
                                                           int Niter) {
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
        T sigma2 = std::pow(sigma0,2);

        //N = Dim(1);
        int N = Signal.n_rows;

        // Reblurred_S = (Signal.*Reblurred)./sigma2;
        Mat<T> Reblurred_S(Signal.n_rows, Signal.n_cols);
        //for (size_t i = 0; i < Signal.n_rows; ++i) {
        //    for (size_t j = 0; j < Signal.n_cols; ++j) {
        //        Reblurred_S(i,j) = (Signal(i,j) * Reblurred(i,j)) / sigma2;
        //    }
        //}
        
        Reblurred_S = Signal % Reblurred / sigma2;

        Mat<T> fODFi;
        Mat<T> Ratio;
        Mat<T> RL_factor(KernelT.n_rows,Reblurred.n_cols);

        // for i = 1:Niter
        for (size_t i = 0; i < Niter; ++i) {
            T sigma2_i;
            // fODFi = fODF;
            fODFi = fODF;

            //Ratio = mBessel_ratio(n_order,Reblurred_S);
            Ratio = mBessel_ratio<T>(n_order,Reblurred_S);

/*
            // ( KernelT*(Reblurred) + eps)
            mat tmp_matrix1(KernelT.n_rows,Reblurred.n_cols);
            tmp_matrix1 =  KernelT * Reblurred;
            for (auto j=0; j < tmp_matrix1.n_rows; ++j) {
                for (auto l=0; l < tmp_matrix1.n_cols; ++l){
                    tmp_matrix1(i,j) += std::numeric_limits<double>::epsilon();
                }
            }
            
            //( Signal.*( Ratio )
            mat tmp_matrix2(Signal.n_rows,Signal.n_cols);
            for (auto j=0; j < Signal.n_rows; ++j) {
                for (auto l=0; l < Signal.n_cols; ++l){
                    tmp_matrix2(j,l) = Signal(j,l) * Ratio(j,l);
                }
            }
            
            // KernelT*( Signal.*( Ratio ) )
            tmp_matrix2 = KernelT * tmp_matrix2;
            
            // RL_factor = KernelT*( Signal.*( Ratio ) )./( KernelT*(Reblurred) + eps);
            for (auto j=0; j < KernelT.n_rows; ++j) {
                for (auto l=0; l < tmp_matrix2.n_cols; ++l){
                    RL_factor(j,l) =  tmp_matrix2(j,l) / tmp_matrix1(j,l);
                }
            }
            */

            RL_factor = KernelT * (Signal % Ratio) / ((KernelT * Reblurred) + std::numeric_limits<double>::epsilon());

            //fODF = fODFi.*RL_factor;
            /*for (auto i = 0; i < fODF.n_rows; ++i) {
                for (auto j = 0; j < fODF.n_cols; ++j) {
                    fODF(i,j) = fODFi(i,j) * RL_factor(i,j);
                }
            }
            */

            fODF = fODFi % RL_factor;
            //% --------------------- Update of variables ------------------------- %
            //Reblurred = Kernel*fODF;
            Reblurred = Kernel * fODF;

            //Reblurred_S = (Signal.*Reblurred)./sigma2;
            /* for (auto i = 0; i < Signal.n_rows; ++i) {
                for (auto j = 0; j < Signal.n_cols; ++j) {
                    Reblurred_S(i,j) = (Signal(i,j) * Reblurred(i,j)) / sigma2;
                }
            } */

            Reblurred_S = (Signal % Reblurred) / sigma2;

            //% -------------------- Estimate the noise level  -------------------- %
            //sigma2_i = (1/N)*sum( (Signal.^2 + Reblurred.^2)/2 - (sigma2.*Reblurred_S).*Ratio, 1)./n_order;
            T sum = 0.0;
   
            #pragma omp parallel for reduction (+:sum)
            for (size_t i = 0; i < Signal.n_rows; ++i) {
                for (size_t j = 0; j < Signal.n_cols; ++j) {
                    sum += (std::pow<T>(Signal(i,j),2) + std::pow<T>(Reblurred(i,j),2)/2 - (sigma2*Reblurred_S(i,j)) * Ratio(i,j));
                }
            }
            sigma2_i =(1.0/N) * sum / n_order;

            //sigma2_i = min((1/10)^2, max(sigma2_i,(1/50)^2)); % robust estimator on the interval sigma = [1/SNR_min, 1/SNR_max],
            sigma2 = std::min<T>(std::pow<T>(1.0/10.0,2),std::max<T>(sigma2_i, std::pow<T>(1.0/50.0,2)));

            //% where SNR_min = 10 and SNR_max = 50
            //sigma2 = repmat(sigma2_i,[N, 1]);
        }

        return fODF;
    }

    
}

#endif

