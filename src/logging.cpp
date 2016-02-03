#include "logging.hpp"
#include <iostream>

#include <cstdlib>
#include <string>
#include <iostream>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup.hpp>
#include <boost/log/support/date_time.hpp>

namespace pfiber {

    void init_logging(bool debug)
    {        
        if (debug) {
        
        boost::log::register_simple_formatter_factory< boost::log::trivial::severity_level, char >("Severity");
        
        // Output message to console
        boost::log::add_console_log(
                                    std::cout,
                                    boost::log::keywords::format = "[%Severity%] -- %Message%"
                                    );

        boost::log::add_file_log (
                                  boost::log::keywords::file_name = "pfiber.log",
                                  boost::log::keywords::format = "[%TimeStamp%] [%Severity%] --  %Message%"
                                  );
       
        boost::log::add_common_attributes();
        
        // Only output message with INFO or higher severity
        boost::log::core::get()->set_filter(
                                            boost::log::trivial::severity >= boost::log::trivial::info
                                            );
        
        } else {
            boost::log::core::get()->set_filter(
                                                boost::log::trivial::severity >= boost::log::trivial::error
                                                );

        }
    }
    
}
