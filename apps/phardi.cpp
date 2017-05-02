/**
* @version		pHARDI v0.3
* @copyright		Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license		GNU/GPL, see LICENSE.txt
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

//#define ARMA_NO_DEBUG

#include "optionparser.hpp"
#include "options.hpp"
#include "constants.hpp"
#include "multi_intravox_fiber_reconstruction.hpp"
#include "config.hpp"
#include <stdlib.h>

#include <plog/Log.h>
#include <plog/Appenders/ColorConsoleAppender.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <armadillo>
#include <arrayfire.h>

enum  optionIndex { UNKNOWN, HELP, READ, DATA, RECONS, MASK, BVECS, BVALS, DEVICE, ODF, PRECISION, OP_RUMBA_NOISE, 
                    OP_RUMBA_ITER, OP_RUMBA_LAMBDA1, OP_RUMBA_LAMBDA2, OP_RUMBA_LAMBDA_CSF, OP_RUMBA_LAMBDA_GM,
                    OP_QBI_LAMBDA, OP_GQI_LAMBDA, OP_GQI_MDDR, OP_DOTR2_LAMBDA, OP_DOTR2_T, OP_DOTR2_EULER, 
                    OP_CSA_LAMBDA, OP_DSI_LMAX, OP_DSI_RES, OP_DSI_RMIN, OP_DSI_LREG, OP_DSI_BOX, OP_DTI_NNLS_TORDER, 
                    ZIP, DEBUG};

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
        if (option.arg != 0 && strtof(option.arg, &endptr)){};
        if (endptr != option.arg && *endptr == 0)
            return option::ARG_OK;
        
        if (msg) printError("Option '", option, "' requires a numeric argument\n");
        return option::ARG_ILLEGAL;
    }
};

const option::Descriptor usage[] =
{
    {UNKNOWN, 0, "", "",option::Arg::None, "USAGE: phardi [options]\n\n"
        "Options:" },
    {HELP, 0,"h", "help",option::Arg::None, "  --help, -h  \tPrint usage and exit." },
    {RECONS, 0,"a","alg",Arg::Required, "  --alg, -a  \tReconstruction method (rumba, dti_nnls, dsi, qbi, gqi_l1, gqi_l2, dotr2, csa)." },
    {READ, 0,"d", "dataread",Arg::NonEmpty, "  --dataread, -d \tData reading method (voxels, slices, volume)." },
    {DATA, 0,"k","data",Arg::Required, "  --data, -k  \tData file." },
    {MASK, 0,"m","mask",Arg::Required, "  --mask, -m  \tBinary mask file." },
    {BVECS, 0,"r","bvecs",Arg::Required, "  --bvecs, -r  \tb-vectors file." },
    {BVALS, 0,"b","bvals",Arg::Required, "  --bvals, -b  \tb-values file." },
    {ODF, 0,"o","odf",Arg::Required, "  --odf, -o  \tOutput path." },
    {PRECISION, 0,"p","presicion",Arg::NonEmpty, "  --precision, -p  \tCalculation precision (float|double)." },

    {DEVICE, 0,"","device",Arg::NonEmpty, "  --device   \tHardware backend: cuda, opencl or cpu (default cuda)." },

    {OP_RUMBA_ITER, 0,"i","rumba-iterations",Arg::Numeric, "  --rumba-iterations \tRUMBA: Iterations performed (default 300)." },
    {OP_RUMBA_LAMBDA1, 0,"","rumba-lambda1",Arg::Numeric, "  --rumba-lambda1 \tRUMBA: Longitudinal diffusivity value, in units of mm^2/s (default 0.0017)." },
    {OP_RUMBA_LAMBDA2, 0,"","rumba-lambda2",Arg::Numeric, "  --rumba-lambda2 \tRUMBA: Radial diffusivity value, in units of mm^2/s (default 0.0003)." },
    {OP_RUMBA_LAMBDA_CSF, 0,"","rumba-lambda-csf",Arg::Numeric, "  --rumba-lambda-csf  \tRUMBA: Diffusivity value in CSF, in units of mm^2/s  (default 0.0030)." },
    {OP_RUMBA_LAMBDA_GM, 0,"","rumba-lambda-gm",Arg::Numeric, "  --rumba-lambda-gm \tRUMBA: Diffusivity value in GM, in units of mm^2/s  (default 0.0007)." },

    {OP_QBI_LAMBDA, 0,"","qbi-lambda",Arg::Numeric, "  --qbi-lambda  \tQBI: Regularization parameter  (default 0.006)." },

    {OP_GQI_LAMBDA, 0,"","gqi-lambda",Arg::Numeric, "  --gqi-lambda  \tGQI: Regularization parameter  (default 1.2)." },
    {OP_GQI_MDDR, 0,"","gqi-meandiffdist ",Arg::Numeric, "  --gqi-meandiffdist  \tGQI: Mean diffusion distance ratio (default 1.2)." },

    {OP_DOTR2_LAMBDA, 0,"","dotr2-lambda",Arg::Numeric, "  --dotr2-lambda  \tDOTR2: Regularization parameter  (default 0.006)." },
    // {OP_DOTR2_T, 0,"","dotr2-t",Arg::Numeric, "  --dotr2-t  \tDOTR2: T value  (default 20.0e-3)." },
    // {OP_DOTR2_EULER, 0,"","dotr2-eulergamma",Arg::Numeric, "  --dotr2-eulergamma  \tDOTR2: Euler Gamma  (default 0.577216)." },

    {OP_CSA_LAMBDA, 0,"","csa-lambda",Arg::Numeric, "  --csa-lambda \tCSA: Regularization parameter  (default 0.006)." },
    
    {OP_DTI_NNLS_TORDER, 0,"","dti_nnls-torder",Arg::Numeric, "  --dti_nnls-torder \tDTI_NNLS: Tensor order  (default 2)." },

    {OP_DSI_LMAX, 0,"","dsi-lmax",Arg::Numeric, "  --dsi-lmax \tDSI: LMAX parameter  (default 10)." },
    {OP_DSI_RES, 0,"","dsi-resolution",Arg::Numeric, "  --dsi-resolution \tDSI: Resolution parameter  (default 35)." },
    {OP_DSI_RMIN, 0,"","dsi-rmin",Arg::Numeric, "  --dsi-rmin \tDSI: RMIN parameter  (default 1)." },
    {OP_DSI_LREG, 0,"","dsi-lreg",Arg::Numeric, "  --dsi-lreg \tDSI: LREG parameter  (default 0.004)." },
    {OP_DSI_BOX, 0,"","dsi-boxhalfwidth",Arg::Numeric, "  --dsi-boxhalfwidth \tDSI: Box half width parameter  (default 5)." },

    {DEBUG, 0,"v","verbose",option::Arg::None, "  --verbose, -v  \tVerbose execution details." },
    {ZIP, 0,"z","compress",option::Arg::None, "  --compress, -z  \tCompress resulting files." },

    {UNKNOWN, 0, "", "",option::Arg::None, "\nExamples:\n"
        " phardi -a rumba -k /data/data.nii.gz -m /data/nodif_brain_mask.nii.gz -r /data/bvecs -b /data/bvals --odf /result/ \n"
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
    using namespace phardi;
    using namespace std::chrono;

    using clk = chrono::high_resolution_clock;

    auto t1 = clk::now();

    argc -= (argc > 0);
    argv += (argc > 0); // skip program name argv[0] if present
    option::Stats stats(usage, argc, argv);
    std::vector<option::Option> options(stats.options_max);
    std::vector<option::Option> buffer(stats.buffer_max);
    option::Parser parse(usage, argc, argv, &options[0], &buffer[0]);

    if (parse.error())
        return 1;

    if (options[HELP] || argc == 0) {
        option::printUsage(std::cout, usage);
        return 0;
    }

    phardi::options opts;

    opts.zip = false;

    if (options[ZIP].count() > 0) {
        opts.zip = true;
    }

    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;

    if (options[DEBUG].count() > 0) {
        plog::init(plog::verbose, &consoleAppender);
    } else {
        plog::init(plog::error, &consoleAppender);
        plog::init(plog::debug, "phardi.log");
    }

    for (option::Option *opt = options.front(); opt; opt = opt->next()) {
        LOG_INFO << "Option: " << opt->name << ": " << opt->arg;
    }

    std::string diffImage = options[DATA].arg;
    std::string bvecsFilename = options[BVECS].arg;
    std::string bvalsFilename = options[BVALS].arg;
    std::string diffBmask = options[MASK].arg;
    std::string ODFfilename = options[ODF].arg;


    std::string recons;

    if (options[RECONS].count() > 0) {
        recons = options[RECONS].arg;
    } else {
        LOG_ERROR << "Reconstruction method not selected.";
        return 1;
    }

    if (opts.zip)
        ODFfilename = ODFfilename + kPathSeparator + "data_odf.nii.gz";
    else
        ODFfilename = ODFfilename + kPathSeparator + "data_odf.nii";

    if (!is_file_exist(diffImage)) {
        LOG_ERROR << "Can't find " << diffImage;
        return 1;
    }

    if (!is_file_exist(bvecsFilename)) {
        LOG_ERROR << "Can't find " << bvecsFilename;
        return 1;
    }

    if (!is_file_exist(bvalsFilename)) {
        LOG_ERROR << "Can't find " << bvalsFilename;
        return 1;
    }

    if (!is_file_exist(diffBmask)) {
        LOG_ERROR << "Can't find " << diffBmask;
        return 1;
    }

    // %% Options
    // opts.reconsMethod = 'rumba_sd'; % Reconstruction Method
    // opts.datreadMethod = 'slices'; % Reading Data
    // opts.saveODF = 1; % Save or not the ODF
    opts.reconsMethod = RUMBA_SD; // Reconstruction Method
    opts.datreadMethod = SLICES;  //Reading Data
    opts.outputDir = options[ODF].arg;

    opts.rumba_sd.add_noise = false;
    opts.rumba_sd.Niter = RUMBA_NITER;
    opts.rumba_sd.lambda1 = RUMBA_LAMBDA1;
    opts.rumba_sd.lambda2 = RUMBA_LAMBDA2;
    opts.rumba_sd.lambda_csf = RUMBA_LAMBDA_CSF;
    opts.rumba_sd.lambda_gm = RUMBA_LAMBDA_GM;

    opts.dsi.lmax = DSI_LMAX;
    opts.dsi.resolution = DSI_RESOLUTION;
    opts.dsi.rmin = DSI_RMIN;
    opts.dsi.lreg = DSI_LREG;
    opts.dsi.boxhalfwidth = DSI_BOXHALFWIDTH;

    opts.qbi.lambda = QBI_LAMBDA;

    opts.gqi.mean_diffusion_distance_ratio = GQI_MEANDIFFDIST;
    opts.gqi.lambda = GQI_LAMBDA;

    opts.dotr2.lambda = DOTR2_LAMBDA;
    opts.dotr2.t = DOTR2_T;
    opts.dotr2.eulerGamma = DOTR2_EULERGAMMA;

    opts.csa.lambda = CSA_LAMBDA;

    opts.dti_nnls.torder = DTI_NNLS_TORDER;

    if (recons == "rumba")
        opts.reconsMethod = RUMBA_SD;
    else if (recons == "dsi")
        opts.reconsMethod = DSI;
    else if (recons == "dotr2")
        opts.reconsMethod = QBI_DOTR2;
    else if (recons == "csa")
        opts.reconsMethod = QBI_CSA;
    else if (recons == "qbi")
        opts.reconsMethod = QBI;
    else if (recons == "gqi_l1")
        opts.reconsMethod = GQI_L1;
    else if (recons == "gqi_l2")
        opts.reconsMethod = GQI_L2;
    else if (recons == "dti_nnls")
        opts.reconsMethod = DTI_NNLS;
    else {
        LOG_ERROR << "Method not recognized. Possible options are: rumba, dti, dsi, dotr2, csa, qbi, gqi_l1, gqi_l2";
        return 0;
    }

    LOG_INFO << "phardi " << VERSION_MAJOR << "." << VERSION_MINOR;
    LOG_INFO << "Start.";

    // Options casting for RUMBA
    if (options[OP_RUMBA_ITER].count() > 0) {
        opts.rumba_sd.Niter = std::stoi(options[OP_RUMBA_ITER].arg);
    }
    if (options[OP_RUMBA_LAMBDA1].count() > 0) {
        opts.rumba_sd.lambda1 = std::stof(options[OP_RUMBA_LAMBDA1].arg);
    }
    if (options[OP_RUMBA_LAMBDA2].count() > 0) {
        opts.rumba_sd.lambda2 = std::stof(options[OP_RUMBA_LAMBDA2].arg);
    }
    if (options[OP_RUMBA_LAMBDA_CSF].count() > 0) {
        opts.rumba_sd.lambda_csf = std::stof(options[OP_RUMBA_LAMBDA_CSF].arg);
    }
    if (options[OP_RUMBA_LAMBDA_GM].count() > 0) {
        opts.rumba_sd.lambda_gm = std::stof(options[OP_RUMBA_LAMBDA_GM].arg);
    }
    if (options[OP_RUMBA_NOISE].count() > 0) {
        opts.rumba_sd.add_noise = true;
    }

    // Options casting for QBI
    if (options[OP_QBI_LAMBDA].count() > 0) {
        opts.qbi.lambda = std::stof(options[OP_QBI_LAMBDA].arg);
    }

    // Options casting for GQI
    if (options[OP_GQI_LAMBDA].count() > 0) {
        opts.gqi.lambda = std::stof(options[OP_GQI_LAMBDA].arg);
    }
    if (options[OP_GQI_MDDR].count() > 0) {
        opts.gqi.mean_diffusion_distance_ratio = std::stof(options[OP_GQI_MDDR].arg);
    }

    // Options casting for DOTR2
    if (options[OP_DOTR2_LAMBDA].count() > 0) {
        opts.dotr2.lambda = std::stof(options[OP_DOTR2_LAMBDA].arg);
    }
    if (options[OP_DOTR2_T].count() > 0) {
        opts.dotr2.t = std::stof(options[OP_DOTR2_T].arg);
    }
    if (options[OP_DOTR2_EULER].count() > 0) {
        opts.dotr2.eulerGamma = std::stof(options[OP_DOTR2_EULER].arg);
    }

    // Options casting for CSA
    if (options[OP_CSA_LAMBDA].count() > 0) {
        opts.csa.lambda = std::stof(options[OP_CSA_LAMBDA].arg);
    }

    // Options casting for DTI_NNLS
    if (options[OP_DTI_NNLS_TORDER].count() > 0) {
        opts.dti_nnls.torder = std::stof(options[OP_DTI_NNLS_TORDER].arg);
    }

    // Options casting for DSI
    if (options[OP_DSI_LMAX].count() > 0) {
        opts.dsi.lmax = std::stoi(options[OP_DSI_LMAX].arg);
    }
    if (options[OP_DSI_RES].count() > 0) {
        opts.dsi.resolution = std::stoi(options[OP_DSI_RES].arg);
    }
    if (options[OP_DSI_RMIN].count() > 0) {
        opts.dsi.rmin = std::stoi(options[OP_DSI_RMIN].arg);
    }
    if (options[OP_DSI_LREG].count() > 0) {
        opts.dsi.lreg = std::stof(options[OP_DSI_LREG].arg);
    }
    if (options[OP_DSI_BOX].count() > 0) {
        opts.dsi.boxhalfwidth = std::stoi(options[OP_DSI_BOX].arg);
    }

    std::string readmethod = "slices";

    if (options[READ].count() > 0) {
        readmethod = std::string(options[READ].arg);
        if (readmethod == "volume") opts.datreadMethod = VOLUME;
        else if (readmethod == "slices") opts.datreadMethod = SLICES;
        else if (readmethod == "voxels") opts.datreadMethod = VOXELS;
        else opts.datreadMethod = SLICES;
    }

    if (opts.reconsMethod == DSI) {
        opts.datreadMethod = VOXELS;
    }

    std::string precision = "float";

    if (options[PRECISION].count() > 0) {
        precision = std::string(options[PRECISION].arg);
    }

    LOG_INFO << "    diffImage     = " << diffImage;
    LOG_INFO << "    bvecsFilename = " << bvecsFilename;
    LOG_INFO << "    bvalsFilename = " << bvalsFilename;
    LOG_INFO << "    diffBmask     = " << diffBmask;
    LOG_INFO << "    ODFfilename   = " << ODFfilename;
    LOG_INFO << "    Precision     = " << precision;

    if (opts.datreadMethod == VOLUME) {
        LOG_INFO << "    Read method   = volume";
    } else if (opts.datreadMethod == SLICES) {
        LOG_INFO << "    Read method   = slices";
    } else if (opts.datreadMethod == VOXELS) {
        LOG_INFO << "    Read method   = voxels";
    }

    if (opts.reconsMethod == DSI && opts.datreadMethod == VOLUME) {
        LOG_INFO << "    Read method not compatible";
        std::cerr << "    Read method not compatible" << std::endl;
        return -1;
    }

    if (opts.reconsMethod == QBI_DOTR2) {
        LOG_INFO << "Configuration details for DOTR2:";
        LOG_INFO << "    Lambda         = " << opts.dotr2.lambda;
        LOG_INFO << "    T              = " << opts.dotr2.t;
        LOG_INFO << "    Euler Gamma    = " << opts.dotr2.eulerGamma;
    }
    else if (opts.reconsMethod == DSI) {
        LOG_INFO << "Configuration details for DSI:";
        LOG_INFO << "    L Max          = " << opts.dsi.lmax;
        LOG_INFO << "    Resolution     = " << opts.dsi.resolution;
        LOG_INFO << "    R Min          = " << opts.dsi.rmin;
        LOG_INFO << "    L Reg          = " << opts.dsi.lreg;
        LOG_INFO << "    Box half width = " << opts.dsi.boxhalfwidth;
    }
    else if (opts.reconsMethod == GQI_L1 || opts.reconsMethod == GQI_L2) {
        LOG_INFO << "Configuration details for GQI:";
        LOG_INFO << "    Mean diffusion distance ratio = " << opts.gqi.mean_diffusion_distance_ratio;
        LOG_INFO << "    Lambda                        = " << opts.gqi.lambda;
    }
    else if (opts.reconsMethod == QBI) {
        LOG_INFO << "Configuration details for QBI:";
        LOG_INFO << "    Lambda                        = " << opts.qbi.lambda;
    }
    else if (opts.reconsMethod == QBI_CSA) {
        LOG_INFO << "Configuration details for CSA:";
        LOG_INFO << "    Lambda                        = " << opts.csa.lambda;
    }
    else if (opts.reconsMethod == RUMBA_SD) {
        LOG_INFO << "Configuration details for RUMBA:";
        LOG_INFO << "    Iterations     = " << opts.rumba_sd.Niter;
        LOG_INFO << "    Lambda 1       = " << opts.rumba_sd.lambda1;
        LOG_INFO << "    Lambda 2       = " << opts.rumba_sd.lambda2;
        LOG_INFO << "    Lambda CSF     = " << opts.rumba_sd.lambda_csf;
        LOG_INFO << "    Lambda GM      = " << opts.rumba_sd.lambda_gm;
    }

    std::string device = "cuda";

    if (options[DEVICE].count() > 0) {
         device = std::string(options[DEVICE].arg);
    }

    int backends = af::getAvailableBackends();

    bool cpu    = backends & AF_BACKEND_CPU;
    bool cuda   = backends & AF_BACKEND_CUDA;
    bool opencl = backends & AF_BACKEND_OPENCL;

    if (device == "cuda" && cuda)
        af::setBackend(AF_BACKEND_CUDA);
    else if (device == "opencl" && opencl)
        af::setBackend(AF_BACKEND_OPENCL);
    else if (device == "cpu" && cpu)
        af::setBackend(AF_BACKEND_CPU);
    else {
        LOG_ERROR << "Device '" << device << "' not available in this computer.";
        LOG_ERROR << "Possible devices:";
        if (cuda)
            LOG_ERROR << "    CUDA";
        if (opencl)
            LOG_ERROR << "    OpenCL";
        if (cpu)
            LOG_ERROR << "    CPU";
        exit(1);
    }

    af::info();

    if (precision == "float")
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
