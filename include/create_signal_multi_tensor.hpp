#ifndef CREATE_SIGNAL_H
#define CREATE_SIGNAL_H

#include "logging.hpp"

#include <iostream>
#include <math.h>
#include <numeric>
#include <armadillo>

namespace pfiber {

    template<typename T>
    arma::Mat<T> RotMatrix(T phi, T theta) {
        using namespace arma;

        //c = pi/180;
        T c = M_PI / 180;

        //phi = phi*c;
        phi = phi * c;

        //theta = theta*c;
        theta = theta * c;

        Mat<T> Rz (3,3);
        Rz(0,0) =   std::cos(phi);   Rz(0,1) = -std::sin(phi);  Rz(0,2) =               0;
        Rz(1,0) =   std::sin(phi);   Rz(1,1) =  std::cos(phi);  Rz(1,2) =               0;
        Rz(2,0) =               0;   Rz(2,1) =              0;  Rz(2,2) =               1;

        Mat<T> Ry (3,3);
        Ry(0,0) =  std::cos(theta);  Ry(0,1) =              0;  Ry(0,2) = std::sin(theta);
        Ry(1,0) =           0;       Ry(1,1) =              1;  Ry(1,2) =               0;
        Ry(2,0) = -std::sin(theta);  Ry(2,1) =              0;  Ry(2,2) = std::cos(theta);

        // R =  Rz*Ry;
        Mat<T> R (3,3);
        R = Rz * Ry;

        //std::cout << "R=" << R << std::endl; 
        return R;
    }

    template<typename T>
    void create_signal_multi_tensor (const arma::Mat<T> & ang,
                                     const arma::Col<T> & f,
                                     const arma::Col<T> & Eigenvalues,
                                     const arma::Col<T> & b,
                                     const arma::Mat<T> & grad,
                                     T S0,
                                     T SNR,
                                     bool add_noise,
                                     arma::Col<T> & S,
                                     arma::Mat<T> & D) {
        using namespace arma;

        // A = diag(Eigenvalues);
        Mat<T> A(3,3);
        A = diagmat(Eigenvalues);

        // S = 0 Not needed

        // Nfibers = length(f);
        size_t Nfibers = f.n_elem;

        // f = f/sum(f)
        size_t sum = std::accumulate(f.begin(), f.end(), 0);

        Col<T> fa = f;
        for (size_t i = 0; i < f.n_elem; ++i)
            fa(i) = fa(i) / sum;

        // for i = 1:Nfibers
        for (size_t i = 0; i < Nfibers; ++i) {

            // phi(i) = ang(i, 1);
            T phi = ang(i,0);

            // heta(i) = ang(i, 2);
            T theta = ang(i,1);

            // R = RotMatrix(phi(i),theta(i));
            Mat<T> R(3,3);
            R = RotMatrix(phi,theta);

            // D = R*A*R';
            D =  R * A * R.t();

            // S = S + f(i)*exp(-b.*diag(grad*D*grad'));
            //     diag(grad*D*grad')

            Mat<T> temp_mat = grad * D * grad.t();
            Col<T> temp = temp_mat.diag();

            for (size_t j = 0; j < S.n_elem; ++j)
                S(j) += fa(i) * std::exp(-b(j)*temp(j));

        }

        // S = S0*S;
        for (size_t i  = 0; i < S.n_elem; ++i)
                S(i) = S(i) * S0;

        if (add_noise) {
            //sigma = S0/SNR;
            T sigma = S0 / SNR;

            // standar_deviation = sigma.*(ones(length(grad),1));
            //med = zeros(length(grad),1);
            //er1 = normrnd(med, standar_deviation);
            //er2 = normrnd(med, standar_deviation);
            Col<T> er1(S.n_elem,fill::randu);
            Col<T> er2(S.n_elem,fill::randu);

            //S = sqrt((S + er1).^2 + er2.^2); % Signal with Rician noise
            for (size_t i  = 0; i < S.n_elem; ++i)
                S(i) = std::sqrt( std::pow(S(i) + er1(i),2) + std::pow(er2(i),2));
        }
    }
}

#endif

