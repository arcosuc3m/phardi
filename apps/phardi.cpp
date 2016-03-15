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

#include "optionparser.hpp"
#include "options.hpp"
#include "constants.hpp"
#include "multi_intravox_fiber_reconstruction.hpp"
#include "config.hpp"

#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <armadillo>


enum  optionIndex { UNKNOWN, HELP, PATH, ODF, PRECISION, NOISE, OP_ITER, OP_LAMBDA1, OP_LAMBDA2, OP_LAMBDA_CSF, OP_LAMBDA_GM,  DEBUG};

struct Arg: public option::Arg
{
    static void printError(const char* msg1, const option::Option& opt, const char* msg2)
    {
        fprintf(stderr, "%s", msg1);
        fwrite(opt.name, opt.namelen, 1, stderr);
        fprintf(stderr, "%s", msg2);
    }
    
    static option::ArgStatus Unknown(const option::Option& option, bool msg)
    {
        if (msg) printError("Unknown option '", option, "'\n");
        return option::ARG_ILLEGAL;
    }
    
    static option::ArgStatus Required(const option::Option& option, bool msg)
    {
        if (option.arg != 0)
            return option::ARG_OK;
        
        if (msg) printError("Option '", option, "' requires an argument\n");
        return option::ARG_ILLEGAL;
    }
    
    static option::ArgStatus NonEmpty(const option::Option& option, bool msg)
    {
        if (option.arg != 0 && option.arg[0] != 0)
            return option::ARG_OK;
        
        if (msg) printError("Option '", option, "' requires a non-empty argument\n");
        return option::ARG_ILLEGAL;
    }
    
    static option::ArgStatus Numeric(const option::Option& option, bool msg)
    {
        char* endptr = 0;
        if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
        if (endptr != option.arg && *endptr == 0)
            return option::ARG_OK;
        
        if (msg) printError("Option '", option, "' requires a numeric argument\n");
        return option::ARG_ILLEGAL;
    }
};

const option::Descriptor usage[] =
{
    {UNKNOWN, 0, "", "",option::Arg::None, "USAGE: pfiber [options]\n\n"
        "Options:" },
    {HELP, 0,"", "help",option::Arg::None, "  --help  \tPrint usage and exit." },
    {PATH, 0,"p","path",Arg::Required, "  --path, -p  \tPath of the input data." },
    {ODF, 0,"o","odf",Arg::Required, "  --odf, -o  \tOutput file name." },
    {PRECISION, 0,"p","presicion",Arg::NonEmpty, "  --precision, -p  \tCalculation precision (float|double)." },
    {OP_ITER, 0,"i","iterations",Arg::Numeric, "  --iterations, -i  \tIterations performed." },
    {OP_LAMBDA1, 0,"l1","lambda1",Arg::Numeric, "  --lambda1, -l1  \tLambda 1 value." },
    {OP_LAMBDA2, 0,"l2","lambda2",Arg::Numeric, "  --lambda2, -l2  \tLambda 2 value." },
    {OP_LAMBDA_CSF, 0,"lc","lambda-csf",Arg::Numeric, "  --lambda-csf, -lc  \tLambda CSF value." },
    {OP_LAMBDA_GM, 0,"lg","lambda-gm",Arg::Numeric, "  --lambda-gm, -lg  \tLambda GM value." },
    {DEBUG, 0,"v","verbose",option::Arg::None, "  --verbose, -v  \tVerbose execution details." },
    {NOISE, 0,"n","noise",option::Arg::None, "  --noise, -n  \tAdd rician noise." },
    {UNKNOWN, 0, "", "",option::Arg::None, "\nExamples:\n"
        "  pfiber --path data/ --odf  data_odf.nii.gz\n"
        "  " },
    {0,0,0,0,0,0}
};

bool is_file_exist(const std::string fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

int main(int argc, char ** argv) {
    using namespace std;
    using namespace pfiber;
    using namespace std::chrono;
    using namespace std::chrono;
    
    using clk = chrono::high_resolution_clock; 
    
    auto t1 = clk::now();
    
    argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
    option::Stats  stats(usage, argc, argv);
    std::vector<option::Option> options(stats.options_max);
    std::vector<option::Option> buffer(stats.buffer_max);
    option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);
    
    if (parse.error())
        return 1;
    
    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return 0;
    }
    
    pfiber::options opts;
    
    if (options[DEBUG].count() > 0) {
       static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
       plog::init(plog::verbose, &consoleAppender);
    } else {
       plog::init(plog::debug, "pfiber.log");
    }

    for (option::Option* opt = options.front(); opt; opt = opt->next()) {
        LOG_INFO << "Option: " << opt->name <<  ": " <<  opt->arg;
    }
    
    std::string inputDir = options[PATH].arg;
    
    std::string diffImage = inputDir + kPathSeparator + "data.nii.gz";
    std::string bvecsFilename = inputDir + kPathSeparator + "bvecs";
    std::string bvalsFilename = inputDir + kPathSeparator + "bvals";
    std::string diffBmask = inputDir + kPathSeparator + "nodif_brain_mask.nii.gz";
    std::string ODFfilename = inputDir + kPathSeparator + options[ODF].arg;

    if (!is_file_exist(diffImage))
    {
        LOG_ERROR << "Can't find " << diffImage;
        return 1;
    }

    if (!is_file_exist(bvecsFilename))
    {
        LOG_ERROR << "Can't find " << bvecsFilename;
        return 1;
    }
    
    if (!is_file_exist(bvalsFilename))
    {
        LOG_ERROR << "Can't find " << bvalsFilename;
        return 1;
    }
    
    if (!is_file_exist(diffBmask))
    {
        LOG_ERROR << "Can't find " << diffBmask;
        return 1;
    }
    
    // %% Options
    // opts.reconsMethod = 'rumba_sd'; % Reconstruction Method
    // opts.datreadMethod = 'slices'; % Reading Data
    // opts.saveODF = 1; % Save or not the ODF
    opts.reconsMethod        = RUMBA_SD; // Reconstruction Method
    opts.datreadMethod       = SLICES;  //Reading Data
    opts.inputDir            = inputDir;
    opts.add_noise           = false;

    opts.rumba_sd.Niter      = NITER;
    opts.rumba_sd.lambda1    = LAMBDA1;
    opts.rumba_sd.lambda2    = LAMBDA2;
    opts.rumba_sd.lambda_csf = LAMBDA_CSF;
    opts.rumba_sd.lambda_gm  = LAMBDA_GM;
 
    LOG_INFO << "pfiber "<< VERSION_MAJOR << "." << VERSION_MINOR;   
    LOG_INFO << "Start.";   


    if (options[OP_ITER].count() > 0) {
        opts.rumba_sd.Niter = std::stof(options[OP_ITER].arg);  
    }
    if (options[OP_LAMBDA1].count() > 0) {
        opts.rumba_sd.lambda1 =  std::stof(options[OP_LAMBDA1].arg);  
    }

    if (options[OP_LAMBDA2].count() > 0) {
        opts.rumba_sd.lambda2 =  std::stof(options[OP_LAMBDA2].arg);  
    }

    if (options[OP_LAMBDA_CSF].count() > 0) {
        opts.rumba_sd.lambda_csf =  std::stof(options[OP_LAMBDA_CSF].arg);  
    }

    if (options[OP_LAMBDA_GM].count() > 0) {
        opts.rumba_sd.lambda_gm =  std::stof(options[OP_LAMBDA_GM].arg);  
    }

    if (options[NOISE].count() > 0) {
        opts.add_noise =  true;
    }
   
    LOG_INFO << "Configuration details:";   
    LOG_INFO << "    Iterations    = " << opts.rumba_sd.Niter;   
    LOG_INFO << "    Lambda 1      = " << opts.rumba_sd.lambda1;   
    LOG_INFO << "    Lambda 2      = " << opts.rumba_sd.lambda2;   
    LOG_INFO << "    Lambda CSF    = " << opts.rumba_sd.lambda_csf;   
    LOG_INFO << "    Lambda GM     = " << opts.rumba_sd.lambda_gm;   
    LOG_INFO << "    diffImage     = " << diffImage;   
    LOG_INFO << "    bvecsFilename = " << bvecsFilename;   
    LOG_INFO << "    bvalsFilename = " << bvalsFilename;   
    LOG_INFO << "    diffBmask     = " << diffBmask;   
    LOG_INFO << "    ODFfilename   = " << ODFfilename;   
    if (options[PRECISION].arg == "float")
    LOG_INFO << "    Precision     = float";   
    else
    LOG_INFO << "    Precision     = double";   

    if (options[PRECISION].arg == "float")
        Multi_IntraVox_Fiber_Reconstruction<float>(diffImage,bvecsFilename,bvalsFilename,diffBmask,ODFfilename,opts);
    else
        Multi_IntraVox_Fiber_Reconstruction<double>(diffImage,bvecsFilename,bvalsFilename,diffBmask,ODFfilename,opts);

    LOG_INFO << "Finalize.";
  
    auto t2 = clk::now();    
    auto diff_milli = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    auto diff_sec = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);
 
    std::cout << "Time = " << diff_sec.count() << "." << diff_milli.count() << " seconds" << endl;

    return 0;
}

