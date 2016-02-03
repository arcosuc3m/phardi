#ifndef CREATE_SIGNAL_H
#define CREATE_SIGNAL_H

#include <armadillo>

namespace pfiber {

arma::mat RotMatrix(double phi, double theta);

void create_signal_multi_tensor (const arma::mat & ang,
                                 const arma::vec & f,
                                 const arma::vec & Eigenvalues,
                                 const arma::vec & b,
                                 const arma::mat & grad,
                                 double S0,
                                 double SNR,
                                 bool add_noise,
                                 arma::vec & S,
                                 arma::mat & D);
}

#endif

