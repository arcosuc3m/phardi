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

#ifndef CREATE_KERNEL_DSI_H
#define CREATE_KERNEL_DSI_H

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1


#include "common.hpp"

#include <plog/Log.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <armadillo>

namespace phardi {

    template <typename T>
    void create_Kernel_for_dsi(const arma::Mat<T> & V,
                                 const arma::Mat<T> & diffGrads,
                                 const arma::Col<T> & diffBvals,
                                 arma::Mat<T> & Kernel,
                                 const phardi::options opts) {

        using namespace arma;

        T PSF;
        Col<T> phi;
        Col<T> theta;
        Cube<T> Sampling_grid;
        Col<T> Laplac2;
        Mat<T> basisV;

        // center_of_image = (opts.dsi.resolution-1)/2 + 1;
        uword center_of_image = (opts.dsi.resolution - 1)/2 + 1;

        //rmin = opts.dsi.rmin;
        uword rmin = opts.dsi.rmin;

        //rmax = opts.dsi.resolution - center_of_image - 1;
        uword rmax = opts.dsi.resolution - center_of_image - 1;

        //r = rmin:(rmax/100):rmax; % radial points that will be used for the ODF radial summation
        Row<T> r = span<Row<T>>(rmin, rmax/100, rmax);

        //rmatrix = repmat(r,[length(V), 1]);
        Mat<T> rmatrix = repmat(r, V.n_elem, 1);

        // for m=1:length(V)
        //     xi(m,:) = center_of_image + r*V(m,2);
        //     yi(m,:) = center_of_image + r*V(m,1);
        //     zi(m,:) = center_of_image + r*V(m,3);
        // end

        Mat<T> xi(0,r.n_elem);
        Mat<T> yi(0,r.n_elem);
        Mat<T> zi(0,r.n_elem);

        for (uword m = 0; m < V.n_rows; ++m) {
            xi.insert_rows(m, center_of_image + r * V(m, 2));
            yi.insert_rows(m, center_of_image + r * V(m, 1));
            zi.insert_rows(m, center_of_image + r * V(m, 3));
        }

        // --- q-space points centered in the new matrix of dimensions Resolution x Resolution x Resolution
        // grad_qspace = round(opts.dsi.boxhalfwidth*diffGrads.*sqrt([diffBvals diffBvals diffBvals]/max(diffBvals)));
        Mat<T> grad_qspace = opts.dsi.boxhalfwidth * diffGrads % sqrt(join_rows(join_rows(diffBvals, diffBvals), diffBvals) / max(diffBvals));

        // qspace = grad_qspace + center_of_image;
        Mat<T> qspace = grad_qspace + center_of_image ;

        // - Computing the point spread function (PSF) for the deconvolution operation
        //   -------------------------------------------------------

        // [PSF, Sampling_grid] = create_mainlobe_PSF(qspace,opts.dsi.resolution);
        create_mainlobe_PSF(qspace, opts.dsi.resolution, PSF, Sampling_grid);


        // Creating Spherical Harmonics
        // Laplac2 = [];
        // for L=0:2:opts.dsi.lmax
        //    for m=-L:L
                //Laplac2 = [Laplac2; (L^2)*(L + 1)^2];
            //end
        // end

        Laplac2.reset() ;
        for (size_t L=0; L < opts.dsi.lmax; L+=2)
        {
            for (size_t m=-(L); m < L; ++m)
            {
                Laplac2 << (pow(L, 2))*pow((L+1), 2);
            }
        }

        //Laplac = diag(Laplac2);
        Mat<T> Laplac = diagmat(Laplac2);

        // basisV = construct_SH_basis (opts.dsi.lmax, V, 2, 'real');
        construct_SH_basis<T>(opts.dsi.lmax, V, 2, "real", theta, phi, basisV) ;

        // Kernel = recon_matrix(basisV,Laplac,opts.dsi.lreg);
        Kernel = recon_matrix<T>(basisV, Laplac, opts.dsi.lreg) ;

        return;
    }
}

#endif
