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

#ifndef CREATE_KERNEL_GQI_H
#define CREATE_KERNEL_GQI_H

#include "common.hpp"

#include <plog/Log.h>
#include <iostream>
#include <math.h>
#include <armadillo>

namespace phardi {

    template <typename T>
    void create_Kernel_for_gqi(const arma::Mat<T> & V,
                                 const arma::Mat<T> & diffGrads,
                                 const arma::Col<T> & diffBvals,
                                 arma::Mat<T> & Kernel,
                                 const phardi::options opts) {

        using namespace arma;

        Mat<T>  b_matrix, x;
        uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
        uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);

        if (opts.reconsMethod == GQI_L1)
        {
            // ODF from the Signal  (L^1)
            // b_matrix = diffGrads(indb1,:).*sqrt(repmat(diffBvals(indb1,:),1,3)*0.018);
            b_matrix = diffGrads.rows(indb1) % sqrt(repmat(diffBvals.rows(indb1), 1, 3) * 0.018) ;

            // Kernel = sinc(V*b_matrix'*mean_diffusion_distance_ratio/pi);
            Kernel = V * trans(b_matrix) * opts.gqi.mean_diffusion_distance_ratio / datum::pi;
            Kernel.transform( [](T val) { return sinc(val); } );
        }
        else if (opts.reconsMethod == GQI_L2)
        {
            // b_matrix = diffGrads(indb1,:).*sqrt(repmat(diffBvals(indb1,:),1,3)*6*0.0040);% I have tested different values and instead of 0.0025,
            b_matrix = diffGrads.rows(indb1) % sqrt(repmat(diffBvals.rows(indb1), 1, 3) * 6.0 * 0.0040) ;

            //x = (V*b_matrix');
            x = V * b_matrix.t();

            //Kernel = (( (2*cos(x))./x.^2 ) + ( (x.^2-2).*sin(x) )./(x+eps).^3);
            Kernel = (((2 * cos(x))/pow(x, 2)) + ((pow(x, 2)-2) % sin(x)) / pow((x+std::numeric_limits<T>::epsilon()), 3)) ;
        }

        return;
    }
}

#endif
