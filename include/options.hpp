#ifndef OPTIONS_H
#define OPTIONS_H

#include <iostream>

namespace fiber {

   enum recons {DOT, SPHDECONV, RUMBA_SD};
   enum datread {VOXELS, SLICES, VOLUME};

   struct options_rumba {
       int Niter;
       double lambda1;
       double lambda2;
       double lambda_csf;
       double lambda_gm;
        
   };
   struct options {
       recons reconsMethod;   // Reconstruction method.
       datread datreadMethod;   // Data reading method.
       std::string inputDir;
       std::string ODFDirscheme;    // Directions scheme for reconstructing ODF.
       bool saveODF;
       options_rumba rumba_sd;
       bool debug;
   }; 
}

#endif

