#include "create_signal_multi_tensor.hpp"
#include "logging.hpp"
#include <iostream>
#include <math.h>
#include <numeric>
#include <armadillo>


namespace pfiber {

    // function R = RotMatrix(phi,theta)
    arma::mat RotMatrix(double phi, double theta) {
        using namespace arma;

        //c = pi/180;
        double c = M_PI / 180;
    
        //phi = phi*c;
        phi = phi * c;
    
        //theta = theta*c;
        theta = theta * c;

        mat Rz (3,3);
        Rz(0,0) =   std::cos(phi);   Rz(0,1) = -std::sin(phi);  Rz(0,2) =               0;
        Rz(1,0) =   std::sin(phi);   Rz(1,1) =  std::cos(phi);  Rz(1,2) =               0;
        Rz(2,0) =               0;   Rz(2,1) =              0;  Rz(2,2) =               1;

        mat Ry (3,3);
        Ry(0,0) =  std::cos(theta);  Ry(0,1) =              0;  Ry(0,2) = std::sin(theta);
        Ry(1,0) =           0;       Ry(1,1) =              1;  Ry(1,2) =               0;
        Ry(2,0) = -std::sin(theta);  Ry(2,1) =              0;  Ry(2,2) = std::cos(theta);

        // R =  Rz*Ry;
        mat R (3,3);
        R = Rz * Ry;
  
        //std::cout << "R=" << R << std::endl; 

        return R;
    }

    // function [S, D] = create_signal_multi_tensor (ang, f, Eigenvalues, b, grad, S0, SNR, add_noise)
    void create_signal_multi_tensor (const arma::mat & ang,
                                     const arma::vec & f,
                                     const arma::vec & Eigenvalues,
                                     const arma::vec & b,
                                     const arma::mat & grad,
                                     double S0,
                                     double SNR,
                                     bool add_noise,
                                     arma::vec & S,
                                     arma::mat & D) {
        using namespace arma;

        // A = diag(Eigenvalues);
        mat A(3,3);
	A = diagmat(Eigenvalues);

        // S = 0 Not needed

        // Nfibers = length(f);
        auto Nfibers = f.n_elem; 

        // f = f/sum(f)
        auto sum = std::accumulate(f.begin(), f.end(), 0);
        
        vec fa = f;
        for (int i = 0; i < f.n_elem; ++i)
            fa(i) = fa(i) / sum;

        // for i = 1:Nfibers
        for (int i = 0; i < Nfibers; ++i) {

            // phi(i) = ang(i, 1);
            auto phi = ang(i,0);
            
            // heta(i) = ang(i, 2);
            auto theta = ang(i,1);

            // R = RotMatrix(phi(i),theta(i));
            mat R(3,3);
            R = RotMatrix(phi,theta);

            // D = R*A*R';
            D =  R * A * R.t();
  
            // S = S + f(i)*exp(-b.*diag(grad*D*grad'));
            //     diag(grad*D*grad')

            mat temp_mat = grad * D * grad.t();
            vec temp = temp_mat.diag();

            for (int j = 0; j < S.n_elem; ++j)
                S(j) +=  fa(i) * std::exp(-b(j)*temp(j));

        }

        // S = S0*S;
        for (int i  = 0; i < S.n_elem; ++i)
                S(i) = S(i) * S0;

        if (add_noise) {
            //sigma = S0/SNR;
            auto sigma = S0 /SNR;

            // standar_deviation = sigma.*(ones(length(grad),1));
            //med = zeros(length(grad),1);
            //er1 = normrnd(med, standar_deviation);
            //er2 = normrnd(med, standar_deviation);
            //S = sqrt((S + er1).^2 + er2.^2); % Signal with Rician noise
        }
    }
}
