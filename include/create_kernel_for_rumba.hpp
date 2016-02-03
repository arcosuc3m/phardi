#ifndef CREATE_KERNEL_RUMBA_H
#define CREATE_KERNEL_RUMBA_H


#include <armadillo>

namespace fiber {

void create_Kernel_for_rumba(const arma::mat & V,
                                const arma::mat & diffGrads,
                                const arma::vec & diffBvals,
                                double lambda1,
                                double lambda2,
                                double lambda_csf,
                                double lambda_gm,
                                arma::mat & kernel);

}

#endif
