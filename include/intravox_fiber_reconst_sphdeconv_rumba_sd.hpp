#ifndef INTRAVOX_H
#define INTRAVOX_H

#include <armadillo>

namespace pfiber {

    arma::mat intravox_fiber_reconst_sphdeconv_rumba_sd(const arma::mat & Signal,
                                                        const arma::mat & Kernel,
                                                        const arma::mat & fODF0,
                                                        int Niter);

    arma::mat mBessel_ratio(double n, const arma::mat & x);
    
}

#endif

