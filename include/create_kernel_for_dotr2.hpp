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
        Mat<T> basisG;
        Mat<uword> Laplac;

        std::vector<T> K_dot_r2_v;
        std::vector<T> K_csa_v;
        std::vector<uword> Laplac2_v;
        sword m, L;
        Col<uword> Laplac2;

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
        indb0 = find(sum(diffGrads,1) == 0);

        //indb1 = find(sum(diffGrads,2) ~= 0);
        indb1 = find(sum(diffGrads,1) != 0);

        // [basisG, thetaG, phiG] = construct_SH_basis (Lmax, diffGrads(indb1,:), 2, 'real');
        construct_SH_basis<T>(Lmax, diffGrads.rows(indb1), 2, "real", thetaG, phiG, basisG);

        // [basisV, thetaV, phiV] = construct_SH_basis (Lmax, V, 2, 'real');
        construct_SH_basis<T>(Lmax, V, 2, "real", thetaV, phiV, basisV);

        // Notice that Table 1 (second row) in the article (Diffusion orientation transform revisited, Neuroimage, 2010) contains typo for L>=6:
        // a term on the denominator was omitted.
        // Use the following corrected result, which contains the omitted term:
        // Vector = [1, 3/32, 15/64, 210/(3*512), 630/(1024*3*5), 13860/(16384*3*5*7)];

        Col<T> Vector = {1.0, 3.0/32.0, 15.0/64.0, 210.0/1536.0, 630.0/15360.0, 13860.0/1720320.0};

        //  K_dot_r2 = []; Laplac2 = []; K_csa = [];


        // for L=0:2:Lmax
        // for m=-L:L
        for (L = 0; L <= Lmax; L+=2) {
            for (m = -(L); m <= L; ++m)
            {
                if (opts.reconsMethod == QBI_DOTR2) {
                    T factor_dot_r2;
                    // -- Precomputation for DOT-R2 method
                    // factor_dot_r2 = ((-1)^(L/2))*Vector(1,L/2 + 1)*(4/pi);
                    factor_dot_r2 = (pow(-1.0, (L / 2))) * Vector(L / 2) * (4.0 / datum::pi);
                    // K_dot_r2 = [K_dot_r2; factor_dot_r2];
                    K_dot_r2_v.push_back(factor_dot_r2);

                } else if (opts.reconsMethod == QBI_CSA) {
                        T factor_csa;
                        // -- Precomputation for CSA-QBI method
                        // if L > 0
                        if (L > 0) {
                            T p1 = 1.0;
                            T p2 = 1.0;
                            // factor_csa = (-1/(8*pi))*((-1)^(L/2))*prod(1:2:(L+1))/prod(2:(L-2));
                            Row<T> reg1 = regspace<Row<T>>(1, 2.0, (L+1.0));
                            Row<T> reg2 = regspace<Row<T>>(2, (L-2.0));
                            p1 = prod(reg1);
                            if (p1 == 0.0) p1 = 1.0;
                            p2 = prod(reg2);
                            if (p2 == 0.0) p2 = 1.0;
                            factor_csa = (-1.0 / (8*datum::pi)) * (pow((-(1.0)), (L/2))) * p1 / p2;
                        } else if (L == 0)
                            // factor_csa = 1/(2*sqrt(pi));
                            factor_csa = 1.0 / (2.0 * std::sqrt(datum::pi)) ;

                        // K_csa = [K_csa; factor_csa];
                        K_csa_v.push_back(factor_csa);
                }
                // -- Precomputation for the spherical harmonic regularization
                Laplac2_v.push_back(pow(L, 2) * pow((L+1), 2));
            }
        }

        K_csa.set_size(K_csa_v.size());
        Laplac2.set_size(Laplac2_v.size());
        K_dot_r2.set_size(K_dot_r2_v.size());

        K_dot_r2 = conv_to<Col<T>>::from(K_dot_r2_v);
        K_csa = conv_to<Col<T>>::from(K_csa_v);
        Laplac2 = conv_to<Col<uword>>::from(Laplac2_v);

        // Laplac = diag(Laplac2);
        Laplac = diagmat(Laplac2);
        // Creating the kernel for the reconstruction
        // Kernel = recon_matrix(basisG,Laplac,Lambda);
        if (opts.reconsMethod == QBI_DOTR2)
            Kernel = recon_matrix<T>(basisG, Laplac, opts.dotr2.lambda);
        else
            Kernel = recon_matrix<T>(basisG, Laplac, opts.csa.lambda);
        return ;

    }
}

#endif
