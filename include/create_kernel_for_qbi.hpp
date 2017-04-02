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

        Mat<T> basisG;
        Mat<uword> Laplac;
        Col<uword> Laplac2;

        Col<T> thetaG, thetaV, phiG, phiV;
        uword Lmax, Nmin;
        sword m, L;

        std::vector<T> K_v;
        std::vector<uword> Laplac2_v;

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
                Mat<T> factor1;

		Col<T> z(1);
                z.zeros();
                factor1 = legendre(L, z) ;
                K_v.push_back(as_scalar(factor1));
                Laplac2_v.push_back(pow(L, 2) * pow((L+1), 2));
            }
        }

        Laplac2.set_size(Laplac2_v.size());
        K.set_size(K_v.size());

        K = conv_to<Col<T>>::from(K_v);
        Laplac2 = conv_to<Col<uword>>::from(Laplac2_v);

        Laplac = diagmat(Laplac2);

        Kernel = recon_matrix<T>(basisG, Laplac, opts.qbi.lambda);
        return;
    }
}

#endif
