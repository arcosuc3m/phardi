#ifndef MULTI_INTRAVOX_H
#define MULTI_INTRAVOX_H

#include "options.hpp"
#include "intravox_fiber_reconst_sphdeconv_rumba_sd.hpp"
#include "logging.hpp"
#include "image.hpp"
#include "create_kernel_for_rumba.hpp"
#include "constants.hpp"

#include <boost/filesystem/path.hpp>
#include <armadillo>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>

namespace pfiber {

    template<typename T>
    void Multi_IntraVox_Fiber_Reconstruction(std::string diffSignalfilename,
                                             std::string diffGradsfilename,
                                             std::string diffBvalsfilename,
                                             std::string diffBmaskfilename,
                                             std::string ODFfilename,
                                             pfiber::options opts
                                             )  {
        using namespace arma;
        using namespace itk;

        Mat<T> V;
        if (opts.ODFDirscheme.length() == 0)
            opts.ODFDirscheme = opts.inputDir + boost::filesystem::path::preferred_separator + "724_shell.txt";


        V.load(opts.ODFDirscheme, arma::raw_ascii);
        BOOST_LOG_TRIVIAL(info) << "Reading V " << opts.ODFDirscheme << " [" << V.n_rows << ", " << V.n_cols << "]";

        Mat<T> diffGrads;
        diffGrads.load(diffGradsfilename, arma::raw_ascii);
        diffGrads = diffGrads.t();

        BOOST_LOG_TRIVIAL(info) << "Reading diffGrads " << diffGradsfilename << " [" << diffGrads.n_rows << ", " << diffGrads.n_cols << "]";

        size_t Ngrad = diffGrads.n_rows;

        Mat<T> diffBvalsM;
        diffBvalsM.load(diffBvalsfilename, arma::raw_ascii);

        Col<T> diffBvals (diffBvalsM.n_elem);
        for (size_t i = 0; i < diffBvalsM.n_elem; ++i) diffBvals(i) = diffBvalsM(0,i);

        BOOST_LOG_TRIVIAL(info) << "Reading diffBvals " << diffBvalsfilename << " [" << diffBvals.n_elem << "]";


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
        Index4DType coord_4d;

        for (int w = 0; w < regionDiff.GetSize()[3]; ++w) {
                Cube<T> temp(regionDiff.GetSize()[0], regionDiff.GetSize()[1], regionDiff.GetSize()[2]);
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

        BOOST_LOG_TRIVIAL(info) << "Diff " << imageDiff;

        Image3DType::Pointer imageMask = Image3DType::New();

        ReadImage<Image3DType>(diffBmaskfilename, imageMask);

        Image3DType::RegionType regionMask = imageMask->GetBufferedRegion();
        Index3DType       index3D      = regionMask.GetIndex();
        Size3DType        size3D       = regionMask.GetSize();
        Spacing3DType     spacing3D    = imageMask->GetSpacing();
        Origin3DType      origin3D     = imageMask->GetOrigin();
        Direction3DType   direction3D  = imageMask->GetDirection();

        BOOST_LOG_TRIVIAL(info) << "Mask size " << regionMask.GetSize();

        Cube<T> Vmask(regionMask.GetSize()[0],regionMask.GetSize()[1],regionMask.GetSize()[2]);

        Index3DType coord_3d;
        for (int x = 0; x < regionMask.GetSize()[0]; ++x) {
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
        BOOST_LOG_TRIVIAL(info) << "created ODF image";
        BOOST_LOG_TRIVIAL(info) << imageODF;

        // %%  =====================================================================

        Mat<T> Kernel (V.n_rows + 2, diffGrads.n_rows);

        BOOST_LOG_TRIVIAL(info) << "Kernel size: " << size(Kernel);
        Kernel.fill(0.0);
        // Kernel = create_Kernel_for_rumba(V, diffGrads, diffBvals, opts.rumba_sd.lambda1, opts.rumba_sd.lambda2, opts.rumba_sd.lambda_csf, opts.rumba_sd.lambda_gm); % Creating Kernel 
        BOOST_LOG_TRIVIAL(info) << "calling create_Kernel_for_rumba";

        create_Kernel_for_rumba<T>(V,diffGrads,diffBvals,opts.rumba_sd.lambda1, opts.rumba_sd.lambda2, opts.rumba_sd.lambda_csf, opts.rumba_sd.lambda_gm, Kernel);

        Mat<T> slicevf_CSF(xdiff,ydiff,fill::zeros);
        Mat<T> slicevf_GM(xdiff,ydiff,fill::zeros);

        Cube<T> globODFslice (xdiff,ydiff,Nd,fill::zeros);

        std::string filenameCSF = opts.inputDir + boost::filesystem::path::preferred_separator + "data-vf_csf.nii.gz";
        std::string filenameGM = opts.inputDir + boost::filesystem::path::preferred_separator + "data-vf_gm.nii.gz";


        size3D[0] = xdiff;   index3D[0] = 0;
        size3D[1] = ydiff;   index3D[1] = 0;
        size3D[2] = zdiff;   index3D[2] = 0;

        Image3DType::Pointer imageCSF = Image3DType::New();
        CreateImage<Image3DType>(imageCSF, size3D, index3D, spacing3D, origin3D, direction3D);

        Image3DType::Pointer imageGM = Image3DType::New();
        CreateImage<Image3DType>(imageGM, size3D, index3D, spacing3D, origin3D, direction3D);

        for (int slice = 0; slice < zdiff; ++slice) {
           BOOST_LOG_TRIVIAL(info) << "Processing slice number " << slice << " of " << zdiff;

           Mat<T> ODF;
           // Imask  = squeeze(spm_slice_vol(Vmask,spm_matrix([0 0 slice]),Vmask.dim(1:2),0));
           //mat Imask = Vmask.slice(slice);

           globODFslice.zeros();
           slicevf_CSF.zeros();
           slicevf_GM.zeros();

           // if sum(Imask(:)) ~= 0
           Cube<T> Idiff(xdiff, ydiff, Ngrad);
           for (int graddir = 0; graddir < Ngrad; ++graddir) {
               // Idiff(:,:,graddir)  = spm_slice_vol(Vdiff(graddir),spm_matrix([0 0 slice]),Vdiff(graddir).dim(1:2),0).*logical(Imask);
               Idiff.slice(graddir) = Vdiff[graddir].slice(slice) % Vmask.slice(slice); // product-wise multiplication
           }

           //inda = find(squeeze(Idiff(:,:,1))>0);
           uvec inda_vec = find(Idiff.slice(0));
           Mat<T> inda = conv_to<Mat<T>>::from(inda_vec);

           //totalNvoxels = prod(size(Imask));
           size_t  totalNvoxels = Vmask.slice(slice).n_elem;

           if (inda.n_elem > 0) {
                   //allIndexes = repmat(inda(:)',[Ngrad 1]); 
                   Mat<T> allIndixes = repmat(inda.t(),Ngrad,1);
                   //allIndixes.save("inda.txt",arma::raw_ascii);

                   // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
                   Mat<T> ind = allIndixes + (totalNvoxels *  repmat(linspace<Mat<T>>(0, Ngrad-1,Ngrad),1,inda.n_elem));
                   Mat<T> diffSignal(Ngrad,inda.n_elem);
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

                             // fODF0 = fODF0/sum(fODF0); % Normalizing ODFs values
                             fODF0.fill(1.0/fODF0.n_elem);

                             BOOST_LOG_TRIVIAL(info) << "calling intravox_fiber_reconst_sphdeconv_rumba_sd";
                             // ODF = Intravox_Fiber_Reconst_sphdeconv_rumba_sd(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter); % Intravoxel Fiber Reconstruction using RUMBA (Canales-Rodriguez, et al 2015)          
                             ODF = intravox_fiber_reconst_sphdeconv_rumba_sd<T>(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter);           

                             for (int i = 0; i < inda.n_elem; ++i) {
                                 slicevf_CSF.at(inda(i)) = ODF(ODF.n_rows - 2 ,i);
                             }

                             for (int i = 0; i < inda.n_elem; ++i) {
                                 slicevf_GM.at(inda(i)) = ODF(ODF.n_rows - 1 ,i);
                             }

                             ODF.resize(ODF.n_rows - 2, ODF.n_cols);
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
                   for (int i = 0; i < ODF.n_rows; ++i) {
                       for (int j = 0; j < ODF.n_cols; ++j) {
                            globODFslice.at(ODFindexes(i,j)) = ODF(i ,j);
                       }
                   }
            }

            for (int i = 0; i < xdiff; ++i) {
                for (int j = 0; j < ydiff; ++j) {
                    Index3DType coord;
                    coord[0] = i; coord[1] = j; coord[2] = slice;
                    imageCSF->SetPixel(coord, slicevf_CSF(i,j));
                }
            }

            for (int i = 0; i < xdiff; ++i) {
                for (int j = 0; j < ydiff; ++j) {
                    Index3DType coord;
                    coord[0] = i; coord[1] = j; coord[2] = slice;
                    imageGM->SetPixel(coord, slicevf_GM(i,j));
                }
            }

            for (int i = 0; i < xdiff; ++i) {
                for (int j = 0; j < ydiff; ++j) {
                    for (int k = 0; k < Nd; ++k) {
                        Index4DType coord;
                        coord[0] = i; coord[1] = j; coord[2] = slice; coord[3] = k;
                        imageODF->SetPixel(coord, globODFslice(i,j,k));
                    }
                }
            }
        }

        BOOST_LOG_TRIVIAL(info) << "writting file " << filenameCSF;
        BOOST_LOG_TRIVIAL(info) << imageCSF;
        WriteImage<Image3DType,NiftiType>(filenameCSF,imageCSF);

        BOOST_LOG_TRIVIAL(info) << "writting file " << filenameGM;
        BOOST_LOG_TRIVIAL(info) <<  imageGM;
        WriteImage<Image3DType,NiftiType>(filenameGM,imageGM);

        BOOST_LOG_TRIVIAL(info) << "writting file " << ODFfilename;
        WriteImage<Image4DType,NiftiType>(ODFfilename,imageODF);
    }
}

#endif

