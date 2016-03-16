# README #



### Requirements ###

* Armadillo 
* ITK 4.9
* GCC 4.9
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
  --help, -h        Print usage and exit.
  --data, -k        Path of the input data.
  --mask, -m        Path of the input mask.
  --bvecs, -r       Path of the input bvecs.
  --bvals, -b       Path of the input bvals.
  --odf, -o         Output path.
  --precision, -p   Calculation precision (float|double).
  --iterations, -i  Iterations performed (default 300).
  --lambda1         Lambda 1 value (default 0.0017).
  --lambda2         Lambda 2 value (default 0.0003).
  --lambda-csf      Lambda CSF value (default 0.0030).
  --lambda-gm       Lambda GM value (default 0.0007).
  --verbose, -v     Verbose execution details.
  --noise, -n       Add rician noise.

Examples:
 phardi -k /data/data.nii.gz -m /data/nodif_brain_mask.nii.gz -r /data/bvecs -b /data/bvals --odf /result/ 
```
