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

#ifndef CREATE_KERNEL_DOTR2_H
#define CREATE_KERNEL_DOTR2_H

#include "common.hpp"

#include <boost/math/special_functions/legendre.hpp>
#include <plog/Log.h>
#include <iostream>
#include <cmath>
#include <armadillo>

namespace phardi {

    template <typename T>
    void create_Kernel_for_dotr2(const arma::Mat<T> & V,
                                 const arma::Mat<T> & diffGrads,
                                 const arma::Col<T> & diffBvals,
                                 arma::Mat<T> & Kernel,
                                 arma::Mat<T> & basisV,
                                 arma::Col<T> & K_dot_r2,
                                 arma::Col<T> & K_csa,
                                 const phardi::options opts) {

        using namespace arma;

        uword Lmax, Nmin;
        uvec indb0, indb1;
        Col<T> thetaG, thetaV, phiG, phiV;
        Mat<T> basisG,  Laplac;
        uword m, L;

        Col<T> Laplac2;

        //  --- real spherical harmonic reconstruction: parameters definition --- %
        // [Lmax Nmin] = obtain_Lmax(diffGrads);
        obtain_Lmax(diffGrads, Lmax, Nmin);

        // if Lmax >= 8
        if (Lmax >= 8)
            // Lmax = 8;
            Lmax = 8;
        // end

        //%display(['The maximum order of the spherical harmonics decomposition is Lmax = ' num2str(Lmax) '.']);
        LOG_INFO << "The maximum order of the spherical harmonics decomposition is Lmax = " << Lmax;

        //indb0 = find(sum(diffGrads,2) == 0);
        indb0 = find(sum(diffGrads,2) == 0);

        //indb1 = find(sum(diffGrads,2) ~= 0);
        indb1 = find(sum(diffGrads,2) != 0);

        // [basisG, thetaG, phiG] = construct_SH_basis (Lmax, diffGrads(indb1,:), 2, 'real');
        construct_SH_basis<T>(Lmax, diffGrads.rows(indb1), 2, "real", thetaG, phiG, basisG);

        // [basisV, thetaV, phiV] = construct_SH_basis (Lmax, V, 2, 'real');
        construct_SH_basis<T>(Lmax, V, 2, "real", thetaV, phiV, basisV);



        // Notice that Table 1 (second row) in the article (Diffusion orientation transform revisited, Neuroimage, 2010) contains typo for L>=6:
        // a term on the denominator was omitted.
        // Use the following corrected result, which contains the omitted term:
        // Vector = [1, 3/32, 15/64, 210/(3*512), 630/(1024*3*5), 13860/(16384*3*5*7)];

        Col<T> Vector = {1, 3/32, 15/64, 210/1536, 630/15360, 13860/1720320};

        //  K_dot_r2 = []; Laplac2 = []; K_csa = [];
        K_csa.reset();
        Laplac2.reset();
        K_dot_r2.reset();

        // for L=0:2:Lmax
        // for m=-L:L
        for (L = 0; L < Lmax; L+=2) {
            for (m=-(L); m<=L; m++)
            {
                switch (opts.reconsMethod) {
                    case QBI_DOTR2:
                        T factor_dot_r2;
                        // -- Precomputation for DOT-R2 method
                        // factor_dot_r2 = ((-1)^(L/2))*Vector(1,L/2 + 1)*(4/pi);
                        factor_dot_r2 = (pow((-(1)), (L/2.0))) * Vector(L/2.0+1) * (4/datum::pi) ;
                        // K_dot_r2 = [K_dot_r2; factor_dot_r2];
                        K_dot_r2 << factor_dot_r2;
                        break;
                    case QBI_CSA:
                        T factor_csa;
                        // -- Precomputation for CSA-QBI method
                        // if L > 0
                        if (L > 0)
                            // factor_csa = (-1/(8*pi))*((-1)^(L/2))*prod(1:2:(L+1))/prod(2:(L-2));
                            factor_csa = (-(1)/(8*datum::pi))*(pow((-(1)), (L/2.0))) * prod(span<Row<T>>(1, 2, (L+1))) / prod(span<Row<T>>(2, (L-2))) ;
                        else if ( L == 0)
                            // factor_csa = 1/(2*sqrt(pi));
                            factor_csa = 1/(2*sqrt(datum::pi)) ;

                        // K_csa = [K_csa; factor_csa];
                        K_csa << factor_csa;
                        break;
                }
                // -- Precomputation for the spherical harmonic regularization
                Laplac2 << (pow(L, 2))*pow((L+1), 2);
            }
        }

        // Laplac = diag(Laplac2);
        Laplac = diagmat(Laplac2);

        // Creating the kernel for the reconstruction
        // Kernel = recon_matrix(basisG,Laplac,Lambda);
        Kernel = recon_matrix<T>(basisG, Laplac, opts.qbi.lambda);

        return ;

    }
}

#endif
