# README #

### Requirements ###

* Armadillo 6.7
* ITK 4.9
* GCC 4.9
* Boost 1.40 (component math: usage of legendre)
* OpenBLAS (for multicore parallelism)
* LaPack

### How do I get set up? ###


```
#!bash

mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-4.9 -DCMAKE_BUILD_TYPE=Release
make
```

### Execution ###


```
#!bash

USAGE: phardi [options]

Options:
Compulsory arguments (You MUST specify ALL arguments):
  --alg, -a           Reconstruction method (rumba, qbi, gqi_l1, gqi_l2, dotr2, csa)
  --data, -k          Data file
  --mask, -m          Binary mask file
  --bvecs, -r         b-vectors file
  --bvals, -b         b-values file
  --odf, -o           Output path

Optional arguments (You may optionally specify one or more of):
  --precision, -p     Calculation precision (float|double)
  --verbose, -v       Verbose execution details
  --help, -h          Print usage and exit

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
  --dotr2-t           T value  (default 20.0e-3).
  --dotr2-eulergamma  Euler Gamma  (default 0.577216).

  CSA
  --csa-lambda        Regularization parameter (default 0.006).



Examples:
 phardi -a rumba -k /data/data.nii.gz -m /data/nodif_brain_mask.nii.gz -r /data/bvecs -b /data/bvals --odf /result/
```
