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

#include "intravox_fiber_reconst_sphdeconv_rumba_sd_gpu.hpp"

namespace phardi {
  
    template<typename T>
    arma::Mat<T> intravox_fiber_reconst_sphdeconv_rumba_sd_gpu(const arma::Mat<T> & Signal,
                                                               const arma::Mat<T> & Kernel,
                                                               const arma::Mat<T> & fODF0,
                                                               int Niter) {
        using namespace arma;

        Mat<T> fODF;

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

        //N = Dim(1);
        int N = Signal.n_rows;

        // Reblurred_S = (Signal.*Reblurred)./sigma2;
        Mat<T> Reblurred_S(Signal.n_rows, Signal.n_cols);
        //for (size_t i = 0; i < Signal.n_rows; ++i) {
        //    for (size_t j = 0; j < Signal.n_cols; ++j) {
        //        Reblurred_S(i,j) = (Signal(i,j) * Reblurred(i,j)) / sigma2;
        //    }
        //}
        
	
        std::vector<T> stl_Signal;
	for (int i = 0; i < Signal.n_rows, ++i)
		for (int j = 0; j < Signal.n_cols, ++j)
			stl_Signal.push_back(Signal(i,j));
	
	intravox_fiber_reconst_sphdeconv_rumba_sd_kernel<T>(stl_Signal, stl_Reblurred, stl_ Kernel, stl_fODF, Niter);

        


        return fODF;
    }
}


