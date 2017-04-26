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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <armadillo>

namespace phardi {

    const double PI                    = 3.14159;

    const int    RUMBA_NITER           = 300;
    const double RUMBA_LAMBDA1         = 1.7e-3;
    const double RUMBA_LAMBDA2         = 0.3e-3;
    const double RUMBA_LAMBDA_CSF      = 3.0e-3;
    const double RUMBA_LAMBDA_GM       =  0.7e-3;

    const arma::uword DSI_LMAX         = 10;
    const arma::uword DSI_RESOLUTION   = 35;
    const arma::uword DSI_RMIN         = 1;
    const double DSI_LREG              = 0.004;
    const arma::uword DSI_BOXHALFWIDTH = 5;

    const double QBI_LAMBDA            = 0.006;

    const double GQI_MEANDIFFDIST      = 1.2;
    const double GQI_LAMBDA            = 1.2;

    const double DOTR2_LAMBDA          = 0.006;
    const double DOTR2_T               = 20.0e-3;
    const double DOTR2_EULERGAMMA      = 0.577216;

    const double CSA_LAMBDA            = 0.006;

    const arma::uword DTI_NNLS_TORDER  = 2;
}

#endif

