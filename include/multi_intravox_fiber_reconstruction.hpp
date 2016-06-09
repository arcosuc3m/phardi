/*
Copyright (c) 2016 
Javier Garcia Blas (fjblas@inf.uc3m.es)
Jose Daniel Garcia Sanchez (josedaniel.garcia@uc3m.es)
/bin/bash: 00: command not found
Erick Canales (ejcanalesr@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and 
associated documentation files (the "Software"), to deal in the Software without restriction, including 
without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the 
following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial 
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN 
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER 
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR 
THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#ifndef MULTI_INTRAVOX_H
#define MULTI_INTRAVOX_H

#include "options.hpp"
#include "intravox_fiber_reconst_sphdeconv_rumba_sd.hpp"
#include "image.hpp"
#include "create_kernel_for_rumba.hpp"
#include "constants.hpp"

#include <plog/Log.h>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>

namespace phardi {

    const char kPathSeparator =
#ifdef _WIN32
                            '\\';
#else
                            '/';
#endif

    template<typename T>
    void Multi_IntraVox_Fiber_Reconstruction(const std::string diffSignalfilename,
                                             const std::string diffGradsfilename,
                                             const std::string diffBvalsfilename,
                                             const std::string diffBmaskfilename,
                                             const std::string ODFfilename,
                                             phardi::options opts
                                             )  {
        using namespace arma;
        using namespace itk;

        Mat<T> V;
        if (opts.ODFDirscheme.length() == 0)
            opts.ODFDirscheme =  "./362_shell_semisphere.txt";


        V.load(opts.ODFDirscheme, arma::raw_ascii);
        LOG_INFO << "Reading V " << opts.ODFDirscheme << " [" << V.n_rows << ", " << V.n_cols << "]";

        Mat<T> diffGrads;
        diffGrads.load(diffGradsfilename, arma::raw_ascii);
        diffGrads = diffGrads.t();

        LOG_INFO << "Reading diffGrads " << diffGradsfilename << " [" << diffGrads.n_rows << ", " << diffGrads.n_cols << "]";

        size_t Ngrad = diffGrads.n_rows;

	//  vector normalization: obtaining unit vectors (0-vectors are preserved) 
	//norm_factor = sqrt(sum(diffGrads.^2,2));
	Col<T> norm_factor = sqrt( sum(pow(diffGrads,2),1) ) + std::numeric_limits<double>::epsilon();
	// diffGrads = diffGrads./repmat(norm_factor + eps,[1 3]);
	diffGrads = diffGrads / repmat(norm_factor, 1, 3);

        Mat<T> diffBvalsM;
        diffBvalsM.load(diffBvalsfilename, arma::raw_ascii);

        Col<T> diffBvals (diffBvalsM.n_elem);
        for (size_t i = 0; i < diffBvalsM.n_elem; ++i) 
		diffBvals(i) = diffBvalsM(0,i);

        LOG_INFO << "Reading diffBvals " << diffBvalsfilename << " [" << diffBvals.n_elem << "]";
	LOG_INFO << "The measurement scheme contains " << Ngrad << " volumes";
	LOG_INFO << "The max b-value is " << diffBvals.max() ;

	// --- volumes where b-value = 0
	// ind_S0 = find(diffBvals.*sum(diffGrads.^2,2) == 0);
	uvec ind_S0_vec = find(diffBvals % sum(pow(diffGrads,2),1) == 0);
        Mat<T> ind_S0 = conv_to<Mat<T>>::from(ind_S0_vec);

	// display(['From the ' num2str(Ngrad) ' volumes, ' num2str(length(ind_S0)) ' are b0 images']);
	LOG_INFO << "From the " << Ngrad << " volumes, " <<  ind_S0.n_elem  << " are b0 images";

	// in rumba_sd the data will be modified/reordered
      	//diffGrads(ind_S0,:) = [];
      	//diffGrads = [0 0 0; diffGrads];
      	//diffBvals(ind_S0) = [];
      	//diffBvals = [0; diffBvals];
      	//size_t Ngrad_mod = diffGrads.n_rows;

        Image4DType::Pointer imageDiff = Image4DType::New();
        ReadImage<Image4DType>(diffSignalfilename, imageDiff);

        Image4DType::RegionType regionDiff = imageDiff->GetBufferedRegion();
        Index4DType       index4D      = regionDiff.GetIndex();
        Size4DType        size4D       = regionDiff.GetSize();
        Spacing4DType     spacing4D    = imageDiff->GetSpacing();
        Origin4DType      origin4D     = imageDiff->GetOrigin();
        Direction4DType   direction4D  = imageDiff->GetDirection();

        // xdiff = Vdiff(1).dim(1);
        // ydiff = Vdiff(1).dim(2);
        // zdiff = Vdiff(1).dim(3);
        int xdiff = regionDiff.GetSize()[0];
        int ydiff = regionDiff.GetSize()[1];
        int zdiff = regionDiff.GetSize()[2];

        int Nd = V.n_rows;

        std::vector<Cube<T>> Vdiff(regionDiff.GetSize()[3]);

	#pragma omp parallel for
        for (int w = 0; w < regionDiff.GetSize()[3]; ++w) {
                Cube<T> temp(regionDiff.GetSize()[0], regionDiff.GetSize()[1], regionDiff.GetSize()[2]);
        	Index4DType coord_4d;
                coord_4d[3]=w;
                for (int x = 0; x < regionDiff.GetSize()[0]; ++x) {
                   coord_4d[0]=x;
                   for (int y = 0; y < regionDiff.GetSize()[1]; ++y) {
                       coord_4d[1]=y;
                        for (int z = 0; z < regionDiff.GetSize()[2]; ++z) {
                            coord_4d[2] = z;
                            temp(x,y,z) = imageDiff->GetPixel(coord_4d);
                        }
                    }
                }
                Vdiff[w] = temp;
        }

        LOG_INFO << "Diff image copied to memory";
        LOG_INFO << "Diff size " << size(Vdiff[0]) << " x " << Vdiff.size() << " volumes";

        Image3DType::Pointer imageMask = Image3DType::New();

        ReadImage<Image3DType>(diffBmaskfilename, imageMask);

        Image3DType::RegionType regionMask = imageMask->GetBufferedRegion();
        Index3DType       index3D      = regionMask.GetIndex();
        Size3DType        size3D       = regionMask.GetSize();
        Spacing3DType     spacing3D    = imageMask->GetSpacing();
        Origin3DType      origin3D     = imageMask->GetOrigin();
        Direction3DType   direction3D  = imageMask->GetDirection();

        LOG_INFO << "Mask size " << regionMask.GetSize();

        Cube<T> Vmask(regionMask.GetSize()[0],regionMask.GetSize()[1],regionMask.GetSize()[2]);

	#pragma omp parallel for
        for (int x = 0; x < regionMask.GetSize()[0]; ++x) {
           Index3DType coord_3d;
           coord_3d[0]=x;
           for (int y = 0; y < regionMask.GetSize()[1]; ++y) {
               coord_3d[1]=y;
                for (int z = 0; z < regionMask.GetSize()[2]; ++z) {
                    coord_3d[2] = z;
                    Vmask(x,y,z) = imageMask->GetPixel(coord_3d);
                }
            }
        }

        // %%  ================== Creating ODF Volume ============================== %

        Image4DType::Pointer imageODF = Image4DType::New();
        Size4DType size4DODF;
        Index4DType index4DODF;

        size4DODF[0] = xdiff;   index4DODF[0] = 0;
        size4DODF[1] = ydiff;   index4DODF[1] = 0;
        size4DODF[2] = zdiff;   index4DODF[2] = 0;
        size4DODF[3] = Nd;      index4DODF[3] = 0;

        CreateImage<Image4DType>(imageODF, size4DODF, index4DODF, spacing4D, origin4D,direction4D);
        LOG_INFO << "created ODF image";
        // LOG_INFO << imageODF;

        // %%  =====================================================================

        Mat<T> Kernel (V.n_rows + 2, diffGrads.n_rows);

        LOG_INFO << "Kernel size: " << size(Kernel);
        Kernel.fill(0.0);
        // Kernel = create_Kernel_for_rumba(V, diffGrads, diffBvals, opts.rumba_sd.lambda1, opts.rumba_sd.lambda2, opts.rumba_sd.lambda_csf, opts.rumba_sd.lambda_gm); % Creating Kernel 
        LOG_INFO << "calling create_Kernel_for_rumba";

        create_Kernel_for_rumba<T>(V,diffGrads,diffBvals,opts.rumba_sd.lambda1, opts.rumba_sd.lambda2, opts.rumba_sd.lambda_csf, opts.rumba_sd.lambda_gm, Kernel, opts);

        std::string filenameCSF = opts.outputDir + kPathSeparator + "data-vf_csf.nii.gz";
        std::string filenameGM = opts.outputDir + kPathSeparator + "data-vf_gm.nii.gz";
        std::string filenameWM = opts.outputDir + kPathSeparator + "data-vf_wm.nii.gz";
        std::string filenameGFA = opts.outputDir + kPathSeparator + "data-vf_gfa.nii.gz";

        size3D[0] = xdiff;   index3D[0] = 0;
        size3D[1] = ydiff;   index3D[1] = 0;
        size3D[2] = zdiff;   index3D[2] = 0;

        Image3DType::Pointer imageCSF = Image3DType::New();
        CreateImage<Image3DType>(imageCSF, size3D, index3D, spacing3D, origin3D, direction3D);

        Image3DType::Pointer imageGM = Image3DType::New();
        CreateImage<Image3DType>(imageGM, size3D, index3D, spacing3D, origin3D, direction3D);

	Image3DType::Pointer imageWM = Image3DType::New();
        CreateImage<Image3DType>(imageWM, size3D, index3D, spacing3D, origin3D, direction3D);

	Image3DType::Pointer imageGFA = Image3DType::New();
        CreateImage<Image3DType>(imageGFA, size3D, index3D, spacing3D, origin3D, direction3D);


	switch (opts.datreadMethod) {
	    case SLICES:
                {
		Mat<T> slicevf_CSF(xdiff,ydiff,fill::zeros);
        	Mat<T> slicevf_GM(xdiff,ydiff,fill::zeros);
        	Mat<T> slicevf_WM(xdiff,ydiff,fill::zeros);
        	Mat<T> slicevf_GFA(xdiff,ydiff,fill::zeros);
        	Row<T> ODF_iso(ydiff,fill::zeros);

        	Cube<T> globODFslice (xdiff,ydiff,Nd,fill::zeros);

		for (int slice = 0; slice < zdiff; ++slice) {
		   LOG_INFO << "Processing slice number " << slice << " of " << zdiff;

		   // Imask  = squeeze(spm_slice_vol(Vmask,spm_matrix([0 0 slice]),Vmask.dim(1:2),0));
		   //mat Imask = Vmask.slice(slice);

		   globODFslice.zeros();
		   slicevf_CSF.zeros();
		   slicevf_GM.zeros();
		   slicevf_WM.zeros();
		   slicevf_GFA.zeros();

		   // if sum(Imask(:)) ~= 0
		   Cube<T> Idiff(xdiff, ydiff, Ngrad);
		   for (int graddir = 0; graddir < Ngrad; ++graddir) {
		       // Idiff(:,:,graddir)  = spm_slice_vol(Vdiff(graddir),spm_matrix([0 0 slice]),Vdiff(graddir).dim(1:2),0).*logical(Imask);
		       Idiff.slice(graddir) = Vdiff[graddir].slice(slice) % Vmask.slice(slice); // product-wise multiplication
		   }

		   // isolating the b0 images                      
		   // S0_est = squeeze( Idiff(:,:,ind_S0) );
		   Cube<T> S0_est (xdiff, ydiff, ind_S0.n_elem);
		   Mat<T> S0_est_M (xdiff, ydiff);

		   for (int i = 0; i < ind_S0.n_elem; ++i) {
			S0_est.slice(i) = Idiff.slice(ind_S0(i));
		    }

		   S0_est_M = mean(S0_est,2); 
		   // if there are several b0 images in the data, we compute the mean value
		   if (ind_S0.n_elem > 1) {
			for (int i = 0; i < ind_S0.n_elem; ++i)
				S0_est.slice(i) = S0_est_M % Vmask.slice(slice);
		   }

		   // reordering the data such that the b0 image appears first (see lines 152-166)
		   // Idiff(:,:,ind_S0) = [];
		   for (int i = ind_S0.n_elem - 1; i >= 0; --i)
			Idiff.shed_slice(ind_S0(i)); 

		   //?? Idiff = cat(3,S0_est,Idiff);
		   Idiff = join_slices(S0_est,Idiff);

		   //  normalize the signal S/S0
		   //for graddir = 1:Ngrad_mod
		   for (int graddir = 0; graddir < Ngrad; ++graddir) {
			//Idiff(:,:,graddir) = squeeze(Idiff(:,:,graddir))./(S0_est + eps);
			Idiff.slice(graddir) = Idiff.slice(graddir) / (S0_est_M + std::numeric_limits<double>::epsilon());
		   }
		   // Repair the signal
		   // Idiff(Idiff>1) = 1;
		   // Idiff(Idiff<0) = 0;
		   Idiff.elem( find(Idiff > 1.0) ).ones();
		   Idiff.elem( find(Idiff < 0.0) ).zeros();
		 
		   //inda = find(squeeze(Idiff(:,:,1))>0);
		   uvec inda_vec = find(Idiff.slice(0) > 0);
		   Mat<T> inda = conv_to<Mat<T>>::from(inda_vec);

		   //totalNvoxels = prod(size(Imask));
		   size_t  totalNvoxels = Vmask.slice(slice).n_elem;

		   Mat<T> ODF;
		   
		   if (inda.n_elem > 0) {
			   //allIndexes = repmat(inda(:)',[Ngrad 1]); 
			   Mat<T> allIndixes = repmat(inda.t(),Ngrad,1);
			   //allIndixes.save("inda.txt",arma::raw_ascii);

			   // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
			   Mat<T> ind = allIndixes + (totalNvoxels *  repmat(linspace<Mat<T>>(0, Ngrad-1,Ngrad),1,inda.n_elem));
			   Mat<T> diffSignal(Ngrad,inda.n_elem);

			   #pragma omp parallel for
			   for (int i = 0; i < Ngrad; ++i) {
			       for (size_t j = 0; j < inda.n_elem; ++j)
				   diffSignal(i,j) = Idiff.at(ind(i,j));
			   }

			   switch (opts.reconsMethod) {
			       case DOT:
				     break;
			       case SPHDECONV:
				     break;
			       case RUMBA_SD:
				     // fODF0 = ones(size(Kernel,2),1);
				     Mat<T> fODF0(Kernel.n_cols,1);
				     T mean_SNR;

				     // fODF0 = fODF0/sum(fODF0); % Normalizing ODFs values
				     fODF0.fill(1.0/fODF0.n_elem);

				     LOG_INFO << "calling intravox_fiber_reconst_sphdeconv_rumba_sd";
				     // ODF = Intravox_Fiber_Reconst_sphdeconv_rumba_sd(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter); % Intravoxel Fiber Reconstruction using RUMBA (Canales-Rodriguez, et al 2015)          
				     ODF = intravox_fiber_reconst_sphdeconv_rumba_sd<T>(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter, mean_SNR);
				     // TODO ODF = intravox_fiber_reconst_sphdeconv_rumba_sd_gpu<T>(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter);
				     LOG_INFO << "Estimated mean SNR = " << mean_SNR;

				     #pragma omp parallel for
				     for (int i = 0; i < inda.n_elem; ++i) {
					 slicevf_CSF.at(inda(i)) = ODF(ODF.n_rows - 2 ,i);
				     }

				     #pragma omp parallel for
				     for (int i = 0; i < inda.n_elem; ++i) {
					 slicevf_GM.at(inda(i)) = ODF(ODF.n_rows - 1 ,i);
				     }


				     //ODF_iso = ODF(end,:) + ODF(end-1,:); 
				     ODF_iso = ODF.row(ODF.n_rows - 1) + ODF.row(ODF.n_rows - 2); 

				     // ODF = ODF(1:end-2,:);
				     ODF.resize(ODF.n_rows - 2, ODF.n_cols);

				     //volvf_WM(inda) = sum(ODF,1);    % Volume fraction of WM 
				     Row<T> tsum = sum(ODF,0);
				     #pragma omp parallel for
				     for (int i = 0; i < inda.n_elem; ++i) {
					 slicevf_WM.at(inda(i)) = tsum(i);
				     }

				     // Adding the isotropic components to the ODF                  
				     ODF = ODF + repmat( ODF_iso / ODF.n_rows , ODF.n_rows, 1 );  
				     ODF = ODF / repmat( sum(ODF,0) + std::numeric_limits<double>::epsilon() , ODF.n_rows,  1 );      

				     // std(ODF,0,1)./( sqrt(mean(ODF.^2,1)) + eps )
				     Row<T>  temp = stddev( ODF, 0, 0 ) / sqrt(mean(pow(ODF,2),0) + std::numeric_limits<double>::epsilon());
				     #pragma omp parallel for
				     for (int i = 0; i < inda.n_elem; ++i) {
					 slicevf_GFA.at(inda(i)) = temp(i);
				     }
				break;
			   }
			   // % Allocating memory to save the ODF
			   // globODFslice = zeros(xdiff,ydiff,Nd);

			   // % Reordering ODF
			   //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
			   Mat<T> allIndexesODF = repmat(inda.t(),ODF.n_rows,1);

			   // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
			   Mat<T> ODFindexes = allIndexesODF + totalNvoxels * repmat(linspace<Mat<T>>(0, ODF.n_rows - 1, ODF.n_rows ),1,inda.n_elem); 

			   // globODFslice(ODFindexes(:)) = ODF(:);
			   #pragma omp parallel for
			   for (int i = 0; i < ODF.n_rows; ++i) {
			       for (int j = 0; j < ODF.n_cols; ++j) {
				    globODFslice.at(ODFindexes(i,j)) = ODF(i ,j);
			       }
			   }
		    }

		    #pragma omp parallel for
		    for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i; coord[2] = slice;
			for (int j = 0; j < ydiff; ++j) {
			    coord[1] = j;
			    imageCSF->SetPixel(coord, slicevf_CSF(i,j));
			}
		    }

		    #pragma omp parallel for
		    for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i;
			coord[2] = slice;
			for (int j = 0; j < ydiff; ++j) {
			    coord[1] = j; 
			    imageGM->SetPixel(coord, slicevf_GM(i,j));
			}
		    }

		    #pragma omp parallel for
		    for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i; coord[2] = slice;
			for (int j = 0; j < ydiff; ++j) {
			    coord[1] = j; 
			    imageWM->SetPixel(coord, slicevf_WM(i,j));
			}
		    }

		    #pragma omp parallel for
		    for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i; coord[2] = slice;
			for (int j = 0; j < ydiff; ++j) {
			    coord[1] = j;
			    imageGFA->SetPixel(coord, slicevf_GFA(i,j));
			}
		    }

		    #pragma omp parallel for
		    for (int i = 0; i < xdiff; ++i) {
			Index4DType coord;
			coord[0] = i; coord[2] = slice; 
			for (int j = 0; j < ydiff; ++j) {
			    coord[1] = j;
			    for (int k = 0; k < Nd; ++k) {
				coord[3] = k;
				imageODF->SetPixel(coord, globODFslice(i,j,k));
			    }
			}
		    }
		}
		}
		break;

	    case VOLUME:
		{
		
		Cube<T> slicevf_CSF(xdiff,ydiff,zdiff,fill::zeros);
        	Cube<T> slicevf_GM(xdiff,ydiff,zdiff,fill::zeros);
        	Cube<T> slicevf_WM(xdiff,ydiff,zdiff,fill::zeros);
        	Cube<T> slicevf_GFA(xdiff,ydiff,zdiff,fill::zeros);
        	Row<T> ODF_iso(ydiff,fill::zeros);

        	std::vector<Cube<T>> globODFslice;
		for (int i = 0; i < Nd; ++i) {
			Cube<T> tmp(xdiff,ydiff,zdiff,fill::zeros);
			globODFslice.push_back(tmp);
		}

		// Imask = logical(Imask);
		Vmask.elem( find( Vmask != 0.0) ).ones();

		if ( accu (Vmask) != 0) {
			Cube<T> S0_est_M (xdiff, ydiff, zdiff,fill::zeros);
			// isolating the b0 images                      
			// S0_est = squeeze( Idiff(:,:,ind_S0) );
		        std::vector<Cube<T>> S0_est;
	
			for (int i = 0; i < ind_S0.n_elem; ++i)
				S0_est.push_back(Vdiff[i]);
			
			// if length(ind_S0) > 1
                   	//    S0_est = mean(S0_est,4).*Imask;
                	// end
		  
			if (ind_S0.n_elem == 1) {
				S0_est_M = S0_est[0];
			} else  {
                                for (int i = 0; i < ind_S0.n_elem; ++i)
                                        S0_est_M = S0_est_M + S0_est[i];
				S0_est_M = S0_est_M / ind_S0.n_elem;
				S0_est_M = S0_est_M % Vmask;
                        }
	
			// reordering the data such that the b0 image appears first (see lines 152-166)
			// Idiff(:,:,:,ind_S0) = [];
			for (int i = 0; i < ind_S0.n_elem; ++i)
				Vdiff[ind_S0(i)].reset();

			for (int graddir = Ngrad - 1; graddir >=0; --graddir) 
				if (Vdiff[graddir].n_elem == 0)  Vdiff.erase(Vdiff.begin() + graddir);

			int pSize = Vdiff.size();

			//Idiff = cat(4,S0_est,Idiff);
			Vdiff.resize( pSize + ind_S0.n_elem);
			for (int i = pSize -1 ; i >= 0; --i)
				Vdiff[i + ind_S0.n_elem] = Vdiff[i];  

			for (int i = 0; i < ind_S0.n_elem; ++i)
				Vdiff[i] = S0_est[i];

			// normalize the signal S/S0
			// for graddir = 1:Ngrad_mod
			#pragma omp parallel for
			for (int graddir = 0; graddir < Ngrad; ++graddir) {
				//Idiff(:,:,:,graddir) = squeeze(Idiff(:,:,:,graddir).* Imask)./(S0_est + eps);
		    		Vdiff[graddir] =  Vdiff[graddir] / (S0_est_M + std::numeric_limits<double>::epsilon());
			}

			// Repair the signal
			// Idiff(Idiff>1) = 1;
			// Idiff(Idiff<0) = 0;
			#pragma omp parallel for
			for (int graddir = 0; graddir < Ngrad; ++graddir) {
				Vdiff[graddir].elem( find(Vdiff[graddir] > 1.0) ).ones();
				Vdiff[graddir].elem( find(Vdiff[graddir] < 0.0) ).zeros();
		 	}
			//inda = find(squeeze(Idiff(:,:,1))>0);
			uvec inda_vec = find( Vmask != 0);
			Mat<T> inda = conv_to<Mat<T>>::from(inda_vec);

			//totalNvoxels = prod(size(Imask));
			size_t  totalNvoxels = Vmask.n_elem;

			//allIndexes = repmat(inda(:)',[Ngrad 1]); 
			Mat<T> allIndixes = repmat(inda.t(),Ngrad,1);
			//allIndixes.save("inda.txt",arma::raw_ascii);

			// diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
			Mat<T> ind = allIndixes + (totalNvoxels *  repmat(linspace<Mat<T>>(0, Ngrad-1,Ngrad),1,inda.n_elem));
			Mat<T> diffSignal(Ngrad,inda.n_elem);

			#pragma omp parallel for
			for (int i = 0; i < Ngrad; ++i) {
			       for (size_t j = 0; j < inda.n_elem; ++j)
				   diffSignal(i,j) = Vdiff[i].at( inda_vec(j)  );  //Vdiff[i].at(ind(i,j));
			}

			Mat<T> ODF;
			switch (opts.reconsMethod) {
			    case DOT:
				  break;
			    case SPHDECONV:
				 break;
			    case RUMBA_SD:
				  // fODF0 = ones(size(Kernel,2),1);
				  Mat<T> fODF0(Kernel.n_cols,1);
				  T mean_SNR;

				  // fODF0 = fODF0/sum(fODF0); % Normalizing ODFs values
				  fODF0.fill(1.0/fODF0.n_elem);

				  LOG_INFO << "calling intravox_fiber_reconst_sphdeconv_rumba_sd";
				  // ODF = Intravox_Fiber_Reconst_sphdeconv_rumba_sd(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter); % Intravoxel Fiber Reconstruction using RUMBA (Canales-Rodriguez, et al 2015)          
				  ODF = intravox_fiber_reconst_sphdeconv_rumba_sd<T>(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter, mean_SNR);
				  // TODO ODF = intravox_fiber_reconst_sphdeconv_rumba_sd_gpu<T>(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter);
				  LOG_INFO << "Estimated mean SNR = " << mean_SNR;


				  #pragma omp parallel for
				  for (int i = 0; i < inda.n_elem; ++i) {
					 slicevf_CSF.at(inda(i)) = ODF(ODF.n_rows - 2 ,i);
				  }

				  #pragma omp parallel for
				  for (int i = 0; i < inda.n_elem; ++i) {
					slicevf_GM.at(inda(i)) = ODF(ODF.n_rows - 1 ,i);
				  }

				  //ODF_iso = ODF(end,:) + ODF(end-1,:); 
				  ODF_iso = ODF.row(ODF.n_rows - 1) + ODF.row(ODF.n_rows - 2); 

				  // ODF = ODF(1:end-2,:);
				  ODF.resize(ODF.n_rows - 2, ODF.n_cols);

				  //volvf_WM(inda) = sum(ODF,1);    % Volume fraction of WM 
				  Row<T> tsum = sum(ODF,0);
				  #pragma omp parallel for
				  for (int i = 0; i < inda.n_elem; ++i) {
					 slicevf_WM.at(inda(i)) = tsum(i);
				  }

				  // Adding the isotropic components to the ODF                  
				  ODF = ODF + repmat( ODF_iso / ODF.n_rows , ODF.n_rows, 1 );  
				  ODF = ODF / repmat( sum(ODF,0) + std::numeric_limits<double>::epsilon() , ODF.n_rows,  1 );      

				  // std(ODF,0,1)./( sqrt(mean(ODF.^2,1)) + eps )
				  Row<T>  temp = stddev( ODF, 0, 0 ) / sqrt(mean(pow(ODF,2),0) + std::numeric_limits<double>::epsilon());
				  #pragma omp parallel for
				  for (int i = 0; i < inda.n_elem; ++i) {
					 slicevf_GFA.at(inda(i)) = temp(i);
				  }
				break;
			}
			// % Allocating memory to save the ODF
			// globODFslice = zeros(xdiff,ydiff,Nd);

			// % Reordering ODF
			//  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
			Mat<T> allIndexesODF = repmat(inda.t(),ODF.n_rows,1);

			// ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
			Mat<T> ODFindexes = allIndexesODF + totalNvoxels * repmat(linspace<Mat<T>>(0, ODF.n_rows - 1, ODF.n_rows ),1,inda.n_elem); 

			// globODFslice(ODFindexes(:)) = ODF(:);
			#pragma omp parallel for
			for (int i = 0; i < ODF.n_rows; ++i) {
			       for (int j = 0; j < ODF.n_cols; ++j) {
				    //globODFslice[i].at(ODFindexes(j)) = ODF(i ,j);
				    globODFslice[i].at(j) = ODF(i ,j);
			       }
			}
		 }

		 #pragma omp parallel for
		 for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i;
			for (int j = 0; j < ydiff; ++j) {
			    coord[1] = j;
			    for (int k = 0; k < zdiff; ++k) {
				T temp = slicevf_CSF(i,j,k);
			    	coord[2] = k;
			    	imageCSF->SetPixel(coord, temp);
			    }
			}
		 }

		 #pragma omp parallel for
		 for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i;
			for (int j = 0; j < ydiff; ++j) {
			    coord[1] = j;
			    for (int k = 0; k < zdiff; ++k) {
				T temp = slicevf_GM(i,j,k);
			    	coord[2] = k;
			    	imageGM->SetPixel(coord, temp);
			    }
			}
		 }

		 #pragma omp parallel for
		 for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i;
			for (int j = 0; j < ydiff; ++j) {
				coord[1] = j;
				for (int k = 0; k < zdiff; ++k) {
			    		T temp = slicevf_WM(i,j,k);
			    		coord[2] = k;
			   		imageWM->SetPixel(coord, temp);
				}
			}
		 }

		 #pragma omp parallel for
		 for (int i = 0; i < xdiff; ++i) {
			Index3DType coord;
			coord[0] = i;
			for (int j = 0; j < ydiff; ++j) {
				coord[1] = j;
				for (int k = 0; k < zdiff; ++k) {
					T temp = slicevf_GFA(i,j,k);
			    		coord[2] = k;
			    		imageGFA->SetPixel(coord, temp);
				}
			}
		 }

		 #pragma omp parallel for
		 for (int n = 0; n < Nd; ++n) {
			Cube<T> tempc = globODFslice[n];
			Index4DType coord;
			coord[3] = n;
		 	for (int i = 0; i < xdiff; ++i) {
				coord[0] = i;
				for (int j = 0; j < ydiff; ++j) {
					coord[1] = j; 
			    		for (int k = 0; k < zdiff; ++k) {
						T temp = tempc(i,j,k);
						coord[2] = k; 
						imageODF->SetPixel(coord, temp);
			    		}
				}
		 	}
		}	
		}	
		break;
	}

        LOG_INFO << "writting file " << filenameCSF;
        // LOG_INFO << imageCSF;
        WriteImage<Image3DType,NiftiType>(filenameCSF,imageCSF);

        LOG_INFO << "writting file " << filenameGM;
        // LOG_INFO <<  imageGM;
        WriteImage<Image3DType,NiftiType>(filenameGM,imageGM);

	LOG_INFO << "writting file " << filenameWM;
        // LOG_INFO <<  imageWM;
        WriteImage<Image3DType,NiftiType>(filenameWM,imageWM);

	LOG_INFO << "writting file " << filenameGFA;
        // LOG_INFO <<  imageGFA;
        WriteImage<Image3DType,NiftiType>(filenameGFA,imageGFA);

        LOG_INFO << "writting file " << ODFfilename;
        WriteImage<Image4DType,NiftiType>(ODFfilename,imageODF);
    }
}

#endif

