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

#ifndef OPTIONS_H
#define OPTIONS_H

#include <iostream>
#include <armadillo>
namespace phardi {

    enum recons {RUMBA_SD, DSI, QBI, GQI_L1, GQI_L2, QBI_DOTR2, QBI_CSA, DTI_NNLS};
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

    struct options_dti_nnls {
       arma::uword torder;
    };

    struct options {
        recons           reconsMethod;   // Reconstruction method.
        datread          datreadMethod;   // Data reading method.
        std::string      outputDir;
        std::string      ODFDirscheme;    // Directions scheme for reconstructing ODF.
        options_rumba    rumba_sd;
        options_dsi      dsi;
        options_gqi      gqi;
        options_qbi      qbi;
        options_csa      csa;
        options_dotr2    dotr2;
        options_dti_nnls dti_nnls;
        bool             zip;
        bool             debug;
    };
}

#endif

