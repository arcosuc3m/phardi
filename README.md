# README #



### Requirements ###

* Boost > 1.54
* Armadillo 
* ITK 4.9
* GCC 4.9
* OpenBLAS (for multicore parallelism)

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
  --path, -p         Path of the input data.
  --odf, -o          Output file name.
  --precision, -p    Calculation precision (float|double).
  --iterations, -i   Iterations performed.
  --lambda1, -l1     Lambda 1 value.
  --lambda2, -l2     Lambda 2 value.
  --lambda-csf, -lc  Lambda CSF value.
  --lambda-gm, -lg   Lambda GM value.
  --verbose, -v      Verbose execution details.
  --noise, -n        Add rician noise.

Examples:
  pfiber --path data/ --odf  data_odf.nii.gz
```
