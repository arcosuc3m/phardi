#ifndef LOGGING_H
#define LOGGING_H

#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>

namespace pfiber {
    void init_logging(bool debug);
}

#endif

