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

#ifndef CREATE_KERNEL_QBI_H
#define CREATE_KERNEL_QBI_H

#include "common.hpp"

#include <boost/math/special_functions/legendre.hpp>
#include <plog/Log.h>
#include <iostream>
#include <cmath>
#include <armadillo>

namespace phardi {

    template <typename T>
    void create_Kernel_for_qbi(const arma::Mat<T> & V,
                                 const arma::Mat<T> & diffGrads,
                                 const arma::Col<T> & diffBvals,
                                 arma::Mat<T> & Kernel,
                                 arma::Mat<T> & basisV,
                                 arma::Col<T> & K,
                                 const phardi::options opts) {

        using namespace arma;


        Mat<T> basisG,  Laplac;
        Col<T> Laplac2;

        Col<T> thetaG, thetaV, phiG, phiV;
        uword Lmax, Nmin;
        T factor1;
        sword m, L;

        obtain_Lmax(diffGrads, Lmax, Nmin);

        if (Lmax >= 8)
        {
            Lmax = 8;
        }

        uvec indb0 = find(sum(diffGrads, 1)==0);
        uvec indb1 = find(sum(diffGrads, 1)!=0);

        construct_SH_basis<T>(Lmax, diffGrads.rows(indb1), 2, "real", thetaG, phiG, basisG);
        construct_SH_basis<T>(Lmax, V, 2, "real", thetaV, phiV, basisV);

        K.reset();
        Laplac2.reset();
        for (L = 0; L <= Lmax; L+=2)
        {
            for (m = -(L); m <= L; ++m)
            {
                factor1 = boost::math::legendre_p(L, 0.0f) ;
                K << factor1;
                Laplac2 << (pow(L, 2))*pow((L+1), 2);
            }
        }
        Laplac = diagmat(Laplac2);

        Kernel = recon_matrix<T>(basisG, Laplac, opts.qbi.lambda);
        return ;

    }
}

#endif
