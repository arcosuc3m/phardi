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

#ifndef OPTIONS_H
#define OPTIONS_H

#include <iostream>
#include <armadillo>
namespace phardi {

    enum recons {RUMBA_SD, DSI, QBI, GQI_L1, GQI_L2, QBI_DOTR2, QBI_CSA};
    enum datread {VOXELS, SLICES, VOLUME};

    struct options_rumba {
        int Niter;
        double lambda1;
        double lambda2;
        double lambda_csf;
        double lambda_gm;
        bool add_noise;
    };
    struct options_dsi {
        arma::uword rmin;
        arma::uword resolution;
        arma::uword lmax;
        double lreg;
        arma::uword boxhalfwidth;
    };

    struct options_qbi {
        double lambda;
    };

    struct options_dotr2 {
        double lambda;
        double t;
        double eulerGamma;
    };

    struct options_csa {
        double lambda;
    };

    struct options_gqi {
        double mean_diffusion_distance_ratio;
        double lambda;
    };

    struct options {
        recons        reconsMethod;   // Reconstruction method.
        datread       datreadMethod;   // Data reading method.
        std::string   outputDir;
        std::string   ODFDirscheme;    // Directions scheme for reconstructing ODF.
        options_rumba rumba_sd;
        options_dsi   dsi;
        options_gqi   gqi;
        options_qbi   qbi;
        options_csa   csa;
        options_dotr2 dotr2;
        bool          debug;

    };
}

#endif

