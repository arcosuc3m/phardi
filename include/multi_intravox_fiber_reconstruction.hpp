#ifndef MULTI_INTRAVOX_H
#define MULTI_INTRAVOX_H

#include "options.hpp"
#include <armadillo>
#include <boost/filesystem/path.hpp>

namespace pfiber {

    void Multi_IntraVox_Fiber_Reconstruction(std::string diffSignal,
                                             std::string diffGrads,
                                             std::string diffBvals,
                                             std::string diffBmask,
                                             std::string ODFfilename,
                                             pfiber::options opts);
   
}

#endif

