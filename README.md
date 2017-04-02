# README #

This is a collection of routines for the analysis of High Angular Resolution Diffusion Imaging (HARDI) data. It is a subset of the code internally used in the FIDMAG Research Foundation (unit of research at the Benito Menni Hospital, Barcelona, Spain), the Cuban Neuroscience Center (Havana, Cuba), and the Medical Imaging Laboratory at Gregorio Marañón Hospital (Madrid, Spain).

List of reconstruction methods currently included:

* Q-Ball Imaging (QBI) 
* Q-Ball Imaging in Constant Solid Angle (CSA-QBI) 
* Revisited version of the DOT method (DOT-R2)
* Diffusion Spectrum Imaging (DSI)
* Robust and Unbiased Model-Based Spherical Deconvolution (RUMBA)


### Requirements ###

The following libraries and compiler are required for compilation:

* GCC (>= 4.9)
* Armadillo (>= 6.7)
* ITK (>= 4.9)
* FFTW (for DSI)
* ArrayFire (>= 3.4.0)

### How do I get set up? ###


```
#!bash

mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-4.9 -DCMAKE_BUILD_TYPE=Release
make
```

### Sample data ###

http://doi.org/10.5281/zenodo.258764

### Execution ###


```
#!bash

USAGE: phardi [options]

Options:
Compulsory arguments (You MUST specify ALL arguments):
  --alg, -a           Reconstruction method (rumba, dsi, qbi, gqi_l1, gqi_l2, dotr2, csa)
  --data, -k          Data file
  --mask, -m          Binary mask file
  --bvecs, -r         b-vectors file
  --bvals, -b         b-values file
  --odf, -o           Output path

Optional arguments (You may optionally specify one or more of):
  --precision, -p     Calculation precision (float|double)
  --verbose, -v       Verbose execution details
  --help, -h          Print usage and exit
  --compress, -z      Compress resulting files
  --device            Hardware backend: cuda, opencl or cpu (default cuda).

Related to each reconstruction method:

  RUMBA:
  --rumba-iterations  Iterations performed (default 300).
  --rumba-lambda1     Longitudinal diffusivity value, in units of mm^2/s (default 0.0017).
  --rumba-lambda2     Radial diffusivity value, in units of mm^2/s (default 0.0003).
  --rumba-lambda-csf  Diffusivity value in CSF, in units of mm^2/s (default 0.0030).
  --rumba-lambda-gm   Diffusivity value in GM, in units of mm^2/s (default 0.0007).
  --rumba-noise       Add rician noise.

  QBI:
  --qbi-lambda        Regularization parameter (default 0.006).

  GQI (L1/L2)
  --gqi-lambda        Regularization parameter (default 1.2).
  --gqi-meandiffdist  Mean diffusion distance ratio (default 1.2).

  DOTR2
  --dotr2-lambda      Regularization parameter (default 0.006).

  CSA
  --csa-lambda        Regularization parameter (default 0.006).

  DSI
  --dsi-lmax          LMAX parameter  (default 10).
  --dsi-resolution    Resolution parameter  (default 35).
  --dsi-rmin          RMIN parameter  (default 1).
  --dsi-lreg          LREG parameter  (default 0.004).
  --dsi-boxhalfwidth  Box half width parameter  (default 5).

Examples:
 phardi -a rumba -k /data/data.nii.gz -m /data/nodif_brain_mask.nii.gz -r /data/bvecs -b /data/bvals --odf /result/
```
