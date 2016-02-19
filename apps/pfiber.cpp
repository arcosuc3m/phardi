#include "optionparser.hpp"
#include "options.hpp"
#include "logging.hpp"
#include "constants.hpp"
#include "multi_intravox_fiber_reconstruction.hpp"

#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <boost/filesystem.hpp>
#include <armadillo>

enum  optionIndex { UNKNOWN, HELP, PATH, ODF, PRECISION, OP_ITER, OP_LAMBDA1, OP_LAMBDA2, OP_LAMBDA_CSF, OP_LAMBDA_GM,  DEBUG};

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
    {PRECISION, 0,"p","presicion",Arg::None, "  --precision, -p  \tCalculation precision (float|double)." },
    {OP_ITER, 0,"i","iterations",Arg::Numeric, "  --iterations, -i  \tIterations performed." },
    {OP_LAMBDA1, 0,"l1","lambda1",Arg::Numeric, "  --lambda1, -l1  \tLambda 1 value." },
    {OP_LAMBDA2, 0,"l2","lambda2",Arg::Numeric, "  --lambda2, -l2  \tLambda 2 value." },
    {OP_LAMBDA_CSF, 0,"lc","lambda-csf",Arg::Numeric, "  --lambda-csf, -lc  \tLambda CSF value." },
    {OP_LAMBDA_GM, 0,"lg","lambda-gm",Arg::Numeric, "  --lambda-gm, -lg  \tLambda GM value." },
    {DEBUG, 0,"v","verbose",option::Arg::None, "  --verbose, -v  \tVerbose execution details." },
    {UNKNOWN, 0, "", "",option::Arg::None, "\nExamples:\n"
        "  pfiber --path data/ --odf  data_odf.nii.gz\n"
        "  " },
    {0,0,0,0,0,0}
};

int main(int argc, char ** argv) {
    using namespace std;
    using namespace pfiber;
    using namespace boost::filesystem;
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
    
    init_logging(options[DEBUG].count() > 0);
    
    BOOST_LOG_TRIVIAL(info) << "Init.";
    
    for (option::Option* opt = options.front(); opt; opt = opt->next()) {
        BOOST_LOG_TRIVIAL(info) << "Option: " << opt->name <<  ": " <<  opt->arg;
    }
    
    std::string inputDir = options[PATH].arg;
    
    std::string diffImage = inputDir + boost::filesystem::path::preferred_separator + "data.nii.gz";
    std::string bvecsFilename = inputDir + boost::filesystem::path::preferred_separator + "bvecs";
    std::string bvalsFilename = inputDir + boost::filesystem::path::preferred_separator + "bvals";
    std::string diffBmask = inputDir + boost::filesystem::path::preferred_separator + "nodif_brain_mask.nii.gz";
    std::string ODFfilename = inputDir + boost::filesystem::path::preferred_separator + options[ODF].arg;

    if (!exists(diffImage))
    {
        BOOST_LOG_TRIVIAL(error) << "Can't find " << diffImage;
        return 1;
    }

    if (!exists(bvecsFilename))
    {
        BOOST_LOG_TRIVIAL(error) << "Can't find " << bvecsFilename;
        return 1;
    }
    
    if (!exists(bvalsFilename))
    {
        BOOST_LOG_TRIVIAL(error) << "Can't find " << bvalsFilename;
        return 1;
    }
    
    if (!exists(diffBmask))
    {
        BOOST_LOG_TRIVIAL(error) << "Can't find " << diffBmask;
        return 1;
    }
    
    
    
    // %% Options
    // opts.reconsMethod = 'rumba_sd'; % Reconstruction Method
    // opts.datreadMethod = 'slices'; % Reading Data
    // opts.saveODF = 1; % Save or not the ODF
    opts.reconsMethod = RUMBA_SD; // Reconstruction Method
    opts.datreadMethod = SLICES;  //Reading Data
    opts.inputDir = inputDir;
      
    opts.rumba_sd.Niter      = NITER;
    opts.rumba_sd.lambda1    = LAMBDA1;
    opts.rumba_sd.lambda2    = LAMBDA2;
    opts.rumba_sd.lambda_csf = LAMBDA_CSF;
    opts.rumba_sd.lambda_gm  = LAMBDA_GM;
 
    BOOST_LOG_TRIVIAL(info) << "Start.";   


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
    BOOST_LOG_TRIVIAL(info) << "Configuration details:";   
    BOOST_LOG_TRIVIAL(info) << "         Iterations = " << opts.rumba_sd.Niter;   
    BOOST_LOG_TRIVIAL(info) << "         Lambda 1   = " << opts.rumba_sd.lambda1;   
    BOOST_LOG_TRIVIAL(info) << "         Lambda 2   = " << opts.rumba_sd.lambda2;   
    BOOST_LOG_TRIVIAL(info) << "         Lambda CSF = " << opts.rumba_sd.lambda_csf;   
    BOOST_LOG_TRIVIAL(info) << "         Lambda GM  = " << opts.rumba_sd.lambda_gm;   

    BOOST_LOG_TRIVIAL(info) << "calling Multi_IntraVox_Fiber_Reconstruction";   
    BOOST_LOG_TRIVIAL(info) << "    diffImage: " << diffImage;   
    BOOST_LOG_TRIVIAL(info) << "    bvecsFilename: " << bvecsFilename;   
    BOOST_LOG_TRIVIAL(info) << "    bvalsFilename: " << bvalsFilename;   
    BOOST_LOG_TRIVIAL(info) << "    diffBmask:" << diffBmask;   
    BOOST_LOG_TRIVIAL(info) << "    ODFfilename:" << ODFfilename;   

    if (options[PRECISION].arg == "float")
        Multi_IntraVox_Fiber_Reconstruction<float>(diffImage,bvecsFilename,bvalsFilename,diffBmask,ODFfilename,opts);
    else
        Multi_IntraVox_Fiber_Reconstruction<double>(diffImage,bvecsFilename,bvalsFilename,diffBmask,ODFfilename,opts);

    BOOST_LOG_TRIVIAL(info) << "Finalize.";
  
    auto t2 = clk::now();    
    auto diff_milli = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    auto diff_sec = std::chrono::duration_cast<std::chrono::seconds>(t2 - t1);
 
    std::cout << "Time = " << diff_sec.count() << "." << diff_milli.count() << " seconds" << endl;

  
    return 0;
}

