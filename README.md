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
* ArrayFire (>= 3.4.0)

### Installation ####

Mandatory packages:

```
sudo apt install libfontconfig1-dev build-essential git cmake libfreeimage-dev cmake-curses-gui freeglut3-dev libinsighttoolkit4.5 
```

Optional packages:

```
sudo apt install libatlas-dev liblapack-dev libblas-dev libopenblas-dev libarpack2-dev liblapacke-dev libatlas3gf-base libatlas3-base opencl-headers
```

Installing OpenCL support and libraries:

- Open source approach:
```
sudo apt install ocl-icd-opencl-dev beignet-opencl-icd opencl-headers mesa-opencl-icd
```

- AMD 

```
sudo apt install ocl-icd-opencl-dev 
```

http://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx


Downloading and Installing ArrayFire:

```
git clone https://github.com/arrayfire/arrayfire.git
cd arrayfire/
git submodule init
git submodule update
mkdir build
cd build/
cmake ..
make -j8
sudo make install
```

### How do I get set up? ###


```
git clone https://github.com/arcosuc3m/phardi
cd phardi/
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 -DCMAKE_BUILD_TYPE=Release
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
  --alg, -a           Reconstruction method (rumba, dsi, qbi, gqi_l1, gqi_l2, dotr2, csa, dti_nnls)
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
  --scheme, -s        ODF spherical representation file path (362 points).

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

  DTI
  --dti_nnls-torder   Tensor order  (default 2).  

Examples:
 phardi -a rumba -k /data/data.nii.gz -m /data/nodif_brain_mask.nii.gz -r /data/bvecs -b /data/bvals --odf /result/
```

### Acknowledgements ###

This work was supported by the EU project ICT 644235 *RePhrase: REfactoring Parallel Heterogeneous Resource-Aware Applications* and project TIN2013-41350-P *Scalable Data Management Techniques for High-End Computing Systems* from the Ministerio de Economía y Competitividad, Spain.
