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

USAGE: pfiber [options]

Options:
  --help             Print usage and exit.
  --data, -k        Path of the input data.
  --mask, -m        Path of the input mask.
  --bvecs, -r       Path of the input bvecs.
  --bvals, -b       Path of the input bvals.
  --odf, -o          Output file name.
  --precision, -p    Calculation precision (float|double).
  --iterations, -i   Iterations performed.
  --lambda1          Lambda 1 value.
  --lambda2          Lambda 2 value.
  --lambda-csf       Lambda CSF value.
  --lambda-gm        Lambda GM value.
  --verbose, -v      Verbose execution details.
  --noise, -n        Add rician noise.

Examples:
  pfiber --path data/ --odf  data_odf.nii.gz
```
