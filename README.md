# README #

Parallel High Angular Resolution Diffusion Imaging (pHARDI) is a toolkit for the GPU/CPU-accelerated reconstruction of intra-voxel reconstruction methods from diffusion Magnetic Resonance Imaging (dMRI) data. It was designed to support multiple linear algebra accelerators in a wide range of devices, such as multi-core GPU devices (both CUDA and OpenCL) or even co-processors, like Intel Xeon Phi. For platforms that do not support any GPU-based accelerator, our solution can also run on multi-core processors (CPU) using highly-tuned linear algebra libraries. We use Armadillo on top of the linear algebra accelerators for providing a common interface and ArrayFire for supporting GPU devices.

List of reconstruction methods:

* Diffusion Tensor Imaging (order-2 or higher) with Symmetric Positive-Definite Constraints (DTI-SPD) (Barmpoutis, 2010)
* Q-Ball Imaging (QBI) (Tuch, 2004; Descoteaux, 2007)
* Q-Ball Imaging in Constant Solid Angle (QBI-CSA) (Aganj, 2010)
* Revisited version of the Diffusion Orientation Transform (DOT-R2) (Canales-Rodríguez, 2010a)
* Generalized Q-sampling Imaging (GQI) (Fang-Cheng, 2010)
* Diffusion Spectrum Imaging (DSI) (Wedeen, 2005; Canales-Rodríguez, 2010b)
* Robust and Unbiased Model-Based Spherical Deconvolution (RUMBA-SD) (Canales-Rodríguez, 2015; Garcia-Blas, 2016)

In the near future we plan to include additional 'state-of-the-art' intra-voxel methods, as well as fiber tracking algorithms.

## Requirements ##

The following libraries and compiler are required for compilation:

* GCC (>= 4.9)
* Armadillo (>= 6.7)
* ITK (>= 4.9)
* ArrayFire (>= 3.4.0)
* Cmake (>= 3.0.0)

## Installation ###

### Using package manager ###

Mandatory packages:

```
sudo apt install libfontconfig1-dev build-essential git cmake libfreeimage-dev cmake-curses-gui freeglut3-dev libinsighttoolkit4.5 
```

Optional packages:

```bash
sudo apt install libatlas-dev liblapack-dev libblas-dev libopenblas-dev libarpack2-dev liblapacke-dev libatlas3gf-base libatlas3-base opencl-headers
```

Installing OpenCL support and libraries:

- Open source approach:
```bash
sudo apt install ocl-icd-opencl-dev beignet-opencl-icd opencl-headers mesa-opencl-icd
```

- AMD 

```bash
sudo apt install ocl-icd-opencl-dev 
```

http://support.amd.com/en-us/kb-articles/Pages/OpenCL2-Driver.aspx


Downloading and Installing ArrayFire:

```bash
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

### Using Superbuild ###

This method will download and cross-compile all dependencies from author's repositories. During this process, all dependencies will be downloaded and compiled under *deps* folder. This method obtains bynaries for: Armadillo (7.950), ITK (4.12.0), Boost (1.64), FFTW (3.3.2), OpenBlas (0.2.19), Lapack (3.7.0), ArrayFire (3.4.2). 

```bash
cmake .. -DUSE_SUPERBUILD=ON -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 -DCMAKE_CXX_COMPILER=/usr/bin/gcc-5 -DCMAKE_BUILD_TYPE=Release
make
```

## How do I get set up? ##

```bash
git clone https://github.com/arcosuc3m/phardi
cd phardi/
mkdir build
cd build
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/g++-5 -DCMAKE_BUILD_TYPE=Release
make 
```

## Sample data ##

http://doi.org/10.5281/zenodo.258764

## Execution ##


```bash
USAGE: phardi [options]

Options:
Compulsory arguments (You MUST specify ALL arguments):
  --alg,   -a         Reconstruction method (dti-spd, qbi, qbi-csa, dotr2, gqi-l1, gqi-l2, dsi, rumba)
  --data,  -k         Data file
  --mask,  -m         Binary mask file
  --bvecs, -r         b-vectors file
  --bvals, -b         b-values file
  --odf,   -o         Output path

Optional arguments (You may optionally specify one or more of):
  --precision, -p     Calculation precision (float|double)
  --verbose,   -v     Verbose execution details
  --help,      -h     Print usage and exit
  --compress,  -z     Compress resulting files
  --device            Hardware backend: cuda, opencl or cpu (default cuda).
  --scheme,    -s     File path to the reconstruction grid (i.e., spherical-mesh) 
                      (default, 362 unit vectors on the hemisphere).

Related to each reconstruction method:

  RUMBA:
  --rumba-iterations  Iterations performed (default 300).
  --rumba-lambda1     Longitudinal diffusivity value, in units of mm^2/s (default 0.0017).
  --rumba-lambda2     Radial diffusivity value, in units of mm^2/s (default 0.0003).
  --rumba-lambda-csf  Diffusivity value in CSF, in units of mm^2/s (default 0.0030).
  --rumba-lambda-gm   Diffusivity value in GM, in units of mm^2/s  (default 0.0007).

  QBI:
  --qbi-lambda        Regularization parameter (default 0.006).

  GQI (L1/L2)
  --gqi-lambda        Regularization parameter (default 1.2).
  --gqi-meandiffdist  Mean diffusion distance ratio (default 1.2).

  DOTR2
  --dotr2-lambda      Regularization parameter (default 0.006).

  CSA
  --qbi-csa-lambda    Regularization parameter (default 0.006).

  DSI
  --dsi-lmax          Maximum spherical harmonic order (default 10).
  --dsi-resolution    Reconstruction grid resolution to compute the propagator (default 35, i.e, 35x35x35).
  --dsi-rmin          RMIN parameter  (default 1).
  --dsi-lreg          Regularization parameter  (default 0.004).
  --dsi-boxhalfwidth  Box half width parameter  (default 5).

  DTI
  --dti-spd-torder    Tensor order  (default 2).  

Examples:
 phardi -a rumba -k /data/data.nii.gz -m /data/nodif_brain_mask.nii.gz -r /data/bvecs -b /data/bvals --odf /result/
```

## Acknowledgements ##

This work has been partially supported through the EU project ICT 644235 *RePhrase: REfactoring Parallel Heterogeneous Resource-Aware Applications* and TIN2016-79637-P *Scalable Data Management Techniques for High-End Computing Systems* from the Ministerio de Economía y Competitividad, Spain, as well as the research project grant PI15/00277 by Instituto de Salud Carlos III (Co-funded by European Regional Development Fund/European Social Fund) "Investing in your future").


## References ##
```
Barmpoutis, A. and Vemuri, B. C. (2010). "A unified framework for estimating
diffusion tensors of any order with symmetric positive-definite constraints".
In Proceedings of ISBI10: IEEE International Symposium on Biomedical Imaging,pages 1385–1388.

Tuch, D.S. (2004). "Q-ball imaging". Magnetic Resonance in Medicine, 52, 1358–1372.

Descoteaux, M., Angelino, E., Fitzgibbons, S., and Deriche, R. (2007). "Regularized, fast, and robust 
analytical q-ball imaging". Magnetic Resonance in Medicine, 58(3), 497–510.

Aganj, I., Lenglet, C., Sapiro, G., Yacoub, E., Ugurbil, K., and Harel, N. (2010).
"Reconstruction of the orientation distribution function in single- and multiple-shell
q-ball imaging within constant solid angle". Magnetic Resonance in Medicine,64,554–566.

Canales-Rodríguez, E.J., Lin, C.P., Iturria-Medina, Y., Yeh, C.H., Cho, K. H., Melie-García, L. (2010a).
"Diffusion orientation transform revisited". NeuroImage,49 (2), 1326–1339.

Yeh, Fang-Cheng, Van Jay Wedeen, and Wen-Yih Isaac Tseng (2010), "Generalized Q-sampling imaging".
Medical Imaging, IEEE Transactions on 29.9: 1626-1635.

Wedeen VJ, Hagmann P, Tseng WY, Reese TG, Weisskoff RM  (2005). "Mapping complex tissue architecture 
with diffusion spectrum magnetic resonance imaging". Magn Reson Med, 54(6), 11377–86.

Canales-Rodríguez, E.J., Iturria-Medina, Y., Alemán-Gómez, Y., Melie-García, L. (2010b). 
"Deconvolution in diffusion spectrum imaging". NeuroImage 50(1): 136-149.

Canales-Rodríguez, E.J., Daducci, A., Sotiropoulos, S.N., Caruyer, E., Aja-Fernández, S., Radua, J., 
Mendizabal, J. M. Y., Iturria-Medina, Y., Melie-García, L., Alemán-Gómez, Y., et al. (2015).
"Spherical deconvolution of multichannel diffusion MRI data with non-Gaussian noise models and spatial 
regularization". PloS one, 10(10).

Garcia-Blas, J., Dolz, M. F., Garcia., J. D., Carretero, J., Daducci, A., Aleman, Y., and 
Canales-Rodriguez, E.J. (2016). "Porting Matlab Applications to High-Performance C++ Codes: 
CPU/GPU-Accelerated Spherical Deconvolution of Diffusion MRI Data", pages 630–643. 
Springer International Publishing, Cham.

Yalamanchili, P., Arshad, U., Mohammed, Z., Garigipati, P., Entschev, P., Kloppenborg, B., Malcolm, J.
and Melonakos, J. (2015). "ArrayFire - A high performance software library for parallel computing with 
an easy-to-use API". Atlanta: AccelerEyes. Retrieved from https://github.com/arrayfire/arrayfire

Sanderson, C. (2016). "Armadillo: An open source C++ linear algebra library for fast prototyping and 
computationally intensive experiments". Journal of Open Source Software, Vol. 1, pp. 26.
Retrieved from http://arma.sourceforge.net/
```
