/**
* @version              pHARDI v0.3
* @copyright            Copyright (C) 2017 Universidad Carlos III de Madrid. All rights reserved.
* @license              GNU/GPL, see LICENSE.txt
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You have received a copy of the GNU General Public License in LICENSE.txt
* also available in <http://www.gnu.org/licenses/gpl.html>.
*
* See COPYRIGHT.txt for copyright notices and details.
*/

#ifndef MULTI_INTRAVOX_H
#define MULTI_INTRAVOX_H

#include "options.hpp"
#include "intravox_fiber_reconst_sphdeconv_rumba_sd.hpp"
#include "image.hpp"
#include "create_kernel_for_rumba.hpp"
#include "create_kernel_for_dsi.hpp"
#include "create_kernel_for_qbi.hpp"
#include "create_kernel_for_gqi.hpp"
#include "create_kernel_for_dotr2.hpp"
#include "constants.hpp"
#include "mirt3D.hpp"

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
                        temp.at(x,y,z) = imageDiff->GetPixel(coord_4d);
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
                    Vmask.at(x,y,z) = imageMask->GetPixel(coord_3d);
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
        Mat<T> basisV;
        Col<T> K;
        Col<T> K_dot_r2;
        Col<T> K_csa;
        Mat<T> qspace;
        Mat<T> xi;
        Mat<T> yi;
        Mat<T> zi;
        Mat<T> rmatrix;

        Kernel.zeros();

        switch (opts.reconsMethod) {
            case RUMBA_SD:
                create_Kernel_for_rumba<T>(V, diffGrads, diffBvals, Kernel, opts);
                LOG_INFO << "Created kernel for RUMBA";
                break;
            case QBI_DOTR2:
            case QBI_CSA:
                create_Kernel_for_dotr2<T>(V, diffGrads, diffBvals, Kernel, basisV, K_dot_r2, K_csa, opts);
                LOG_INFO << "Created kernel for CSA/DOTR2";
                break;
            case DSI:
                create_Kernel_for_dsi<T>(V, diffGrads, diffBvals, Kernel, basisV, qspace, xi, yi, zi, rmatrix, opts);
                LOG_INFO << "Created kernel for DSI";
                break;
            case QBI:
                create_Kernel_for_qbi<T>(V, diffGrads, diffBvals, Kernel, basisV, K, opts);
                LOG_INFO << "Created kernel for QBI";
                break;
            case GQI_L1:
            case GQI_L2:
                create_Kernel_for_gqi<T>(V, diffGrads, diffBvals, Kernel, opts);
                LOG_INFO << "Created kernel for GQI_L1/GQI_L2";
                break;
        }

        std::string filenameCSF;
        std::string filenameGM;
        std::string filenameWM;
        std::string filenameGFA;

        if (opts.zip) {
            filenameCSF = opts.outputDir + kPathSeparator + "data-vf_csf.nii.gz";
            filenameGM  = opts.outputDir + kPathSeparator + "data-vf_gm.nii.gz";
            filenameWM  = opts.outputDir + kPathSeparator + "data-vf_wm.nii.gz";
            filenameGFA = opts.outputDir + kPathSeparator + "data-vf_gfa.nii.gz";
        }
        else {
            filenameCSF = opts.outputDir + kPathSeparator + "data-vf_csf.nii";
            filenameGM  = opts.outputDir + kPathSeparator + "data-vf_gm.nii";
            filenameWM  = opts.outputDir + kPathSeparator + "data-vf_wm.nii";
            filenameGFA = opts.outputDir + kPathSeparator + "data-vf_gfa.nii";
        }

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
            case VOXELS:
            {
                Mat<T> ODF;
                Cube<T> Idiff(xdiff, ydiff, Ngrad);
                Cube<T> globODFslice (xdiff,ydiff,Nd,fill::zeros);

                // for slice = 1:Vdiff(1).dim(3) %%%%% === This loop could be parallelized
                for (uword slice = 0; slice < zdiff; ++slice) {
                    //   clear ODF;
                    ODF.clear();

                    //   disp(['Processing slice number ' num2str(slice) ' of ' num2str(Vdiff(1).dim(3))]);
                    LOG_INFO << "Processing slice number " << slice << " of " << zdiff;

                    //   Imask  = squeeze(spm_slice_vol(Vmask,spm_matrix([0 0 slice]),Vmask.dim(1:2),0)); % Reading Binary Mask slice
                    Mat<T> Imask = Vmask.slice(slice);
                    //   if sum(Imask(:)) ~= 0
                    if (accu(vectorise(Imask)) != 0) {
                        // Loading Diffusion Data
                        // for graddir = 1:Ngrad_mod

                        Cube<T> Idiff(xdiff, ydiff, Ngrad);
                        for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                            // Idiff(:,:,graddir)  = spm_slice_vol(Vdiff(graddir),spm_matrix([0 0 slice]),Vdiff(graddir).dim(1:2),0).*logical(Imask);
                            Idiff.slice(graddir) = Vdiff[graddir].slice(slice) % Vmask.slice(slice); // product-wise multiplication
                        }

                        // Voxel indexes
                        // inda = find(squeeze(Idiff(:,:,1))>0);
                        uvec inda_vec = find(Idiff.slice(0) > 0);
                        Mat<uword> inda = conv_to<Mat<uword>>::from(inda_vec);

                        // totalNvoxels = prod(size(Imask)); % Total Number of voxels in the slice
                        uword totalNvoxels = Imask.n_rows * Imask.n_cols;

                        // allIndexes = repmat(inda(:)',[Ngrad 1]); % Indexes
                        Mat<uword> allIndexes = repmat(inda.t(),Ngrad,1);

                        // Diffusion signal matrix evaluated in the indexes inside brain mask
                        // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
                        Mat<T> diffSignal(Ngrad,inda.n_elem);
                        Mat<uword> ind = allIndexes + (totalNvoxels *  repmat(linspace<Mat<uword>>(0, Ngrad-1,Ngrad),1,inda.n_elem));
#pragma omp parallel for
                        for (uword j = 0; j < inda.n_elem; ++j)
                            for (uword i = 0; i < Ngrad; ++i)
                                diffSignal.at(i,j) = Idiff.at(ind(i,j));


                        ODF.resize(rmatrix.n_rows, inda.n_elem);
                        switch (opts.reconsMethod) {
                            case DSI: {
                                // --- Signal in the 3D image
                                // Smatrix = SignalMatrixBuilding(qspace,diffSignal,opts.dsi.resolution)
                                Mat<T> tempSignal;

                                uvec indb0, indb1;
                                // indb0 = find(sum(diffGrads,2) == 0);
                                indb0 = find(sum(diffGrads, 1) == 0);

                                // indb1 = find(sum(diffGrads,2) ~= 0);
                                indb1 = find(sum(diffGrads, 1) != 0);

                                // tempSignal = diffSignal(indb1,:);
                                Mat<T> tempS0;
                                // if length(indb0)>1
                                if (indb0.n_elem > 1)
                                    // tempS0 = repmat(mean(diffSignal(indb0,:)),[size(diffSignal,1) 1]); % Signal from B0
                                    tempS0 = repmat(mean(diffSignal.rows(indb0)), size(diffSignal, 0), 1);
                                    // else
                                else
                                    // tempS0 = repmat(diffSignal(indb0,:),[size(diffSignal,1) 1]); % Signal from B0
                                    tempS0 = repmat(diffSignal.rows(indb0), size(diffSignal, 0), 1);

                                // tempSignal = diffSignal./tempS0;
                                tempSignal = diffSignal / tempS0;

                                // for indvox = 1:length(inda)
#pragma omp parallel for
                                for (uword indvox = 0; indvox < inda.n_elem; ++indvox) {

                                    // Smatrix = SignalMatrixBuilding_Volume(qspace,tempSignal(:,indvox),opts.dsi.resolution);
                                    Mat<T> vox(tempSignal.n_rows,1);
                                    vox.col(0) = tempSignal.col(indvox);

                                    Cube<T> Smatrix = SignalMatrixBuilding_Volume(qspace, vox, opts.dsi.resolution);

                                    // --- DSI: PDF computation via fft

                                    // Pdsi = real(fftshift(fftn(ifftshift(Smatrix))));
                                    Cube<T> Pdsi = real(fftshift3D(fft3D(ifftshift3D(Smatrix))));

                                    // Pdsi(Pdsi<0)=0;
                                    Pdsi.elem(find(Pdsi < 0.0)).zeros();

                                    // Pdsi_int = mirt3D_mexinterp(Pdsi,xi,yi,zi);
                                    Mat<T> Pdsi_int = mirt3D_Function(Pdsi, xi, yi, zi);

                                    // Pdsi_int(Pdsi_int<0) = 0;
                                    Pdsi_int.elem(find(Pdsi_int < 0.0)).zeros();

                                    // Pdsi_int = Pdsi_int./sum(Pdsi_int(:));
                                    Pdsi_int = Pdsi_int / sum(vectorise(Pdsi_int));

                                    // --- Numerical ODF-DSI reconstruction

                                    // ODFvox = sum(Pdsi_int.*(rmatrix.^2),2);
                                    Col<T> ODFvox = sum(Pdsi_int % pow(rmatrix,2),1);

                                    // ODFvox = ODFvox/sum(ODFvox);
                                    ODFvox = ODFvox / sum(ODFvox);

                                    // ODF(:,indvox) = ODFvox;
                                    ODF.col(indvox) = ODFvox;

                                    // --- Representing ODF-DSI in terms of spherical harmonics
                                }
                                // Smoothing
                                // sphE_dsi = Kernel*ODF;
                                Mat<T> sphE_dsi = Kernel * ODF;

                                // ODF = abs(basisV*sphE_dsi);
                                ODF = abs(basisV*sphE_dsi);

                                // ODF = ODF./repmat(sum(ODF),[size(ODF,1) 1]);
                                ODF = ODF / repmat(sum(ODF), size(ODF,0), 1);

                            }
                                break;
                        }

                        // Allocating memory to save the ODF
                        // globODFslice = zeros(xdiff,ydiff,Nd);

                        // % Reordering ODF
                        //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                        Mat<uword> allIndexesODF = repmat(inda.t(),ODF.n_rows,1);

                        // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                        Mat<uword> ODFindexes = allIndexesODF + totalNvoxels * repmat(linspace<Mat<uword>>(0, ODF.n_rows - 1, ODF.n_rows ),1,inda.n_elem);

                        // globODFslice(ODFindexes(:)) = ODF(:);
#pragma omp parallel for
                        for (uword j = 0; j < ODF.n_cols; ++j) {
                            for (uword i = 0; i < ODF.n_rows; ++i) {
                                globODFslice.at(ODFindexes(i,j)) = ODF.at(i ,j);
                            }
                        }

                    }
#pragma omp parallel for
                    for (uword i = 0; i < xdiff; ++i) {
                        Index4DType coord;
                        coord[0] = i; coord[2] = slice;
                        for (uword j = 0; j < ydiff; ++j) {
                            coord[1] = j;
                            for (uword k = 0; k < Nd; ++k) {
                                coord[3] = k;
                                imageODF->SetPixel(coord, globODFslice.at(i,j,k));
                            }
                        }
                    }
                }

            }
                break;
            case SLICES:
            {
                Mat<T> slicevf_CSF(xdiff,ydiff,fill::zeros);
                Mat<T> slicevf_GM(xdiff,ydiff,fill::zeros);
                Mat<T> slicevf_WM(xdiff,ydiff,fill::zeros);
                Mat<T> slicevf_GFA(xdiff,ydiff,fill::zeros);
                Row<T> ODF_iso(ydiff,fill::zeros);

                Cube<T> globODFslice (xdiff,ydiff,Nd,fill::zeros);

                for (uword slice = 0; slice < zdiff; ++slice) {
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
                    for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                        // Idiff(:,:,graddir)  = spm_slice_vol(Vdiff(graddir),spm_matrix([0 0 slice]),Vdiff(graddir).dim(1:2),0).*logical(Imask);
                        Idiff.slice(graddir) = Vdiff[graddir].slice(slice) % Vmask.slice(slice); // product-wise multiplication
                    }

                    // isolating the b0 images
                    // S0_est = squeeze( Idiff(:,:,ind_S0) );
                    Cube<T> S0_est (xdiff, ydiff, ind_S0.n_elem);
                    Mat<T> S0_est_M (xdiff, ydiff);

                    for (uword i = 0; i < ind_S0.n_elem; ++i) {
                        S0_est.slice(i) = Idiff.slice(ind_S0(i));
                    }

                    S0_est_M = mean(S0_est,2);
                    // if there are several b0 images in the data, we compute the mean value
                    if (ind_S0.n_elem > 1) {
#pragma omp parallel for
                        for (uword i = 0; i < ind_S0.n_elem; ++i)
                            S0_est.slice(i) = S0_est_M % Vmask.slice(slice);
                    }

                    // reordering the data such that the b0 image appears first (see lines 152-166)
                    // Idiff(:,:,ind_S0) = [];
                    for (sword i = ind_S0.n_elem - 1; i >= 0; --i)
                        Idiff.shed_slice(ind_S0(i));

                    //?? Idiff = cat(3,S0_est,Idiff);
                    Idiff = join_slices(S0_est,Idiff);

                    //  normalize the signal S/S0
                    //for graddir = 1:Ngrad_mod
                    for (uword graddir = 0; graddir < Ngrad; ++graddir) {
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
                    Mat<uword> inda = conv_to<Mat<uword>>::from(inda_vec);

                    //totalNvoxels = prod(size(Imask));
                    uword  totalNvoxels = Vmask.slice(slice).n_elem;

                    Mat<T> ODF;

                    if (inda.n_elem > 0) {
                        //allIndexes = repmat(inda(:)',[Ngrad 1]);
                        Mat<uword> allIndexes = repmat(inda.t(),Ngrad,1);
                        //allIndixes.save("inda.txt",arma::raw_ascii);

                        // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
                        Mat<uword> ind = allIndexes + (totalNvoxels *  repmat(linspace<Mat<uword>>(0, Ngrad-1,Ngrad),1,inda.n_elem));
                        Mat<T> diffSignal(Ngrad,inda.n_elem);

#pragma omp parallel for
                        for (uword j = 0; j < inda.n_elem; ++j)
                            for (uword i = 0; i < Ngrad; ++i)
                                diffSignal.at(i,j) = Idiff.at(ind.at(i,j));

                        switch (opts.reconsMethod) {
                            case QBI_DOTR2:
                            {
                                uvec indb0, indb1;
                                //indb0 = find(sum(diffGrads,2) == 0);
                                indb0 = find(sum(diffGrads, 1) == 0);

                                //indb1 = find(sum(diffGrads,2) ~= 0);
                                indb1 = find(sum(diffGrads, 1) != 0);

                                //tempSignal = diffSignal(indb1,:);
                                Mat<T> tempSignal = diffSignal.rows(indb1);
                                Mat<T> tempS0;
                                //if length(indb0)>1
                                if (indb0.n_elem > 1)
                                    //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                    //else
                                else
                                    //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);

                                // tempVar = -log(tempSignal./tempS0)./repmat(diffBvals(indb1),[1 size(tempSignal,2)]);
                                Mat<T> tempVar = -log(tempSignal / tempS0) / repmat(diffBvals.rows(indb1), 1, size(tempSignal, 1));

                                // ADC = abs(tempVar);
                                Mat<T> ADC = abs(tempVar);

                                //Dt_nonoise = ADC*opts.qbi_dotr2.t;
                                Mat<T> Dt_nonoise = ADC * opts.dotr2.t;

                                //Signal_profile = opts.qbi_dotr2.eulerGamma + log(1./Dt_nonoise);
                                Mat<T> Signal_profile = opts.dotr2.eulerGamma + log(1.0 / Dt_nonoise);

                                //coeff0 = Kernel*Signal_profile;
                                Mat<T> coeff0 = Kernel * Signal_profile;

                                // coeff0 = coeff0./repmat(coeff0(1,:),[size(coeff0,1) 1]);
                                coeff0 = coeff0 / repmat(coeff0.row(0), size(coeff0, 0), 1);

                                //K_dot_r2(1) = (4/pi)*(2*sqrt(pi))/16;
                                K_dot_r2(0) = (4 / datum::pi) * (2 * std::sqrt(datum::pi)) / 16.0;

                                //ss = coeff0.*repmat(K_dot_r2,[1 size(coeff0,2)]);
                                Mat<T> ss = coeff0 % repmat(K_dot_r2, 1, size(coeff0, 1));

                                // ODF = basisV*ss;
                                ODF = basisV * ss;

                                // ODF = ODF - repmat(min(ODF),size(V,1),1);
                                ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                                // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                                ODF = ODF / repmat(sum(ODF,0), size(V, 0), 1);
                            }
                                break;
                            case QBI_CSA:
                            {
                                uvec indb0, indb1;
                                //indb0 = find(sum(diffGrads,2) == 0);
                                indb0 = find(sum(diffGrads, 1)==0);

                                //indb1 = find(sum(diffGrads,2) ~= 0);
                                indb1 = find(sum(diffGrads, 1)!=0);

                                //tempSignal = diffSignal(indb1,:);
                                Mat<T> tempSignal = diffSignal.rows(indb1) ;
                                Mat<T> tempS0;
                                //if length(indb0)>1
                                if (indb0.n_elem > 1)
                                    //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                    //else
                                else
                                    //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);

                                // Signal_profile = log(-log(tempSignal./tempS0));
                                Mat <T> Signal_profile = log(-log(tempSignal / tempS0));

                                // coeff0 = Kernel*Signal_profile;
                                Mat<T> coeff0 = Kernel * Signal_profile;

                                //coeff0(1,:) = ones(1,size(coeff0,2));
                                coeff0.row(0) = ones<Row<T>>(size(coeff0, 1));

                                // ss = coeff0.*repmat(K_csa,[1 size(coeff0,2)]);
                                Mat<T> ss = coeff0 % repmat (K_csa ,1 , size(coeff0, 1));

                                // ODF = basisV*ss;
                                ODF = basisV * ss;

                                // ODF = ODF - repmat(min(ODF),size(V,1),1);
                                ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                                // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                                ODF = ODF/repmat(sum(ODF,0), size(V, 0), 1);
                            }
                                break;
                            case GQI_L1:
                            {
                                uvec indb0, indb1;
                                //indb0 = find(sum(diffGrads,2) == 0);
                                indb0 = find(sum(diffGrads, 1)==0);

                                //indb1 = find(sum(diffGrads,2) ~= 0);
                                indb1 = find(sum(diffGrads, 1)!=0);

                                //tempSignal = diffSignal(indb1,:);
                                Mat<T> tempSignal = diffSignal.rows(indb1) ;
                                Mat<T> tempS0;
                                //if length(indb0)>1
                                if (indb0.n_elem > 1)
                                    //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                    //else
                                else
                                    //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);
                                //end
                                //tempSignal = tempSignal./tempS0;
                                tempSignal = tempSignal / tempS0;

                                //ODF = Kernel*tempSignal;
                                ODF = Kernel * tempSignal;

                                //ODF = ODF - repmat(min(ODF),size(V,1),1);
                                ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);

                                //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                                ODF = ODF/repmat(sum(ODF, 0), size(V, 0), 1);
                            }
                                break;
                            case GQI_L2:
                            {
                                uvec indb0, indb1;
                                //indb0 = find(sum(diffGrads,2) == 0);
                                indb0 = find(sum(diffGrads, 1)==0);

                                //indb1 = find(sum(diffGrads,2) ~= 0);
                                indb1 = find(sum(diffGrads, 1)!=0);

                                //tempSignal = diffSignal(indb1,:);
                                Mat<T> tempSignal = diffSignal.rows(indb1) ;
                                Mat<T> tempS0;
                                //if length(indb0)>1
                                if (indb0.n_elem > 1)
                                    //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                    //else
                                else
                                    //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                    tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);
                                //end
                                //tempSignal = tempSignal./tempS0;
                                tempSignal = tempSignal / tempS0;

                                //ODF = Kernel*tempSignal*opts.gqi_l2.lambda^3/pi;
                                ODF = Kernel * tempSignal * pow(opts.gqi.lambda, 3) / datum::pi;

                                //ODF = ODF - repmat(min(ODF),size(V,1),1);
                                ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                                //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                                ODF = ODF/repmat(sum(ODF,0), size(V, 0), 1);
                            }
                                break;
                            case QBI:
                            {
                                uvec indb0, indb1;
                                Mat<T> coeff, ss;

                                // indb0 = find(sum(diffGrads,2) == 0);
                                indb0 = find(sum(diffGrads, 1)==0);
                                // indb1 = find(sum(diffGrads,2) ~= 0);
                                indb1 = find(sum(diffGrads, 1)!=0);

                                // coeff = Kernel*diffSignal(indb1,:);
                                coeff = Kernel * diffSignal.rows(indb1);
                                // ss = coeff.*repmat(K,[1 size(coeff,2)]);
                                ss = coeff % repmat(K, 1, size(coeff,1));

                                // ODF = basisV*ss;
                                ODF = basisV * ss;

                                // ODF = ODF - repmat(min(ODF),size(V,1),1);
                                ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                                // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                                ODF = ODF/repmat(sum(ODF,0), size(V, 0), 1);
                            }
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
                                for (uword i = 0; i < inda.n_elem; ++i) {
                                    slicevf_CSF.at(inda(i)) = ODF.at(ODF.n_rows - 2 ,i);
                                }

#pragma omp parallel for
                                for (uword i = 0; i < inda.n_elem; ++i) {
                                    slicevf_GM.at(inda(i)) = ODF.at(ODF.n_rows - 1 ,i);
                                }


                                //ODF_iso = ODF(end,:) + ODF(end-1,:);
                                ODF_iso = ODF.row(ODF.n_rows - 1) + ODF.row(ODF.n_rows - 2);

                                // ODF = ODF(1:end-2,:);
                                ODF.resize(ODF.n_rows - 2, ODF.n_cols);

                                //volvf_WM(inda) = sum(ODF,1);    % Volume fraction of WM
                                Row<T> tsum = sum(ODF,0);
#pragma omp parallel for
                                for (uword i = 0; i < inda.n_elem; ++i) {
                                    slicevf_WM.at(inda.at(i)) = tsum.at(i);
                                }

                                // Adding the isotropic components to the ODF
                                ODF = ODF + repmat( ODF_iso / ODF.n_rows , ODF.n_rows, 1 );
                                ODF = ODF / repmat( sum(ODF,0) + std::numeric_limits<double>::epsilon() , ODF.n_rows,  1 );

                                // std(ODF,0,1)./( sqrt(mean(ODF.^2,1)) + eps )
                                Row<T>  temp = stddev( ODF, 0, 0 ) / sqrt(mean(pow(ODF,2),0) + std::numeric_limits<double>::epsilon());
#pragma omp parallel for
                                for (uword i = 0; i < inda.n_elem; ++i) {
                                    slicevf_GFA.at(inda.at(i)) = temp.at(i);
                                }
                                break;
                        }
                        // % Allocating memory to save the ODF
                        // globODFslice = zeros(xdiff,ydiff,Nd);

                        // % Reordering ODF
                        //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                        Mat<uword> allIndexesODF = repmat(inda.t(),ODF.n_rows,1);

                        // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                        Mat<uword> ODFindexes = allIndexesODF + totalNvoxels * repmat(linspace<Mat<uword>>(0, ODF.n_rows - 1, ODF.n_rows ),1,inda.n_elem);

                        // globODFslice(ODFindexes(:)) = ODF(:);
#pragma omp parallel for
                        for (uword j = 0; j < ODF.n_cols; ++j) {
                            for (uword i = 0; i < ODF.n_rows; ++i) {
                                globODFslice.at(ODFindexes.at(i,j)) = ODF.at(i ,j);
                            }
                        }
                    }
                    if (opts.reconsMethod == RUMBA_SD) {
#pragma omp parallel for
                        for (uword j = 0; j < ydiff; ++j) {
                            Index3DType coord;
                            coord[1] = j; coord[2] = slice;
                            for (uword i = 0; i < xdiff; ++i) {
                                coord[0] = i;
                                imageCSF->SetPixel(coord, slicevf_CSF.at(i,j));
                            }
                        }

#pragma omp parallel for
                        for (uword j = 0; j < ydiff; ++j) {
                            Index3DType coord;
                            coord[1] = j; coord[2] = slice;
                            for (uword i = 0; i < xdiff; ++i) {
                                coord[0] = i;
                                imageGM->SetPixel(coord, slicevf_GM.at(i,j));
                            }
                        }

#pragma omp parallel for
                        for (uword j = 0; j < ydiff; ++j) {
                            Index3DType coord;
                            coord[1] = j; coord[2] = slice;
                            for (uword i = 0; i < xdiff; ++i) {
                                coord[0] = i;
                                imageWM->SetPixel(coord, slicevf_WM.at(i,j));
                            }
                        }

#pragma omp parallel for
                        for (uword j = 0; j < ydiff; ++j) {
                            Index3DType coord;
                            coord[1] = j; coord[2] = slice;
                            for (uword i = 0; i < xdiff; ++i) {
                                coord[0] = i;
                                imageGFA->SetPixel(coord, slicevf_GFA.at(i,j));
                            }
                        }
                    }
#pragma omp parallel for
                    for (uword i = 0; i < xdiff; ++i) {
                        Index4DType coord;
                        coord[0] = i; coord[2] = slice;
                        for (uword j = 0; j < ydiff; ++j) {
                            coord[1] = j;
                            for (uword k = 0; k < Nd; ++k) {
                                coord[3] = k;
                                imageODF->SetPixel(coord, globODFslice.at(i,j,k));
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
                for (uword i = 0; i < Nd; ++i) {
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

                    for (uword i = 0; i < ind_S0.n_elem; ++i)
                        S0_est.push_back(Vdiff[i]);

                    // if length(ind_S0) > 1
                    //    S0_est = mean(S0_est,4).*Imask;
                    // end

                    if (ind_S0.n_elem == 1) {
                        S0_est_M = S0_est[0];
                    } else  {
                        for (uword i = 0; i < ind_S0.n_elem; ++i)
                            S0_est_M = S0_est_M + S0_est[i];
                        S0_est_M = S0_est_M / ind_S0.n_elem;
                        S0_est_M = S0_est_M % Vmask;
                    }

                    // reordering the data such that the b0 image appears first (see lines 152-166)
                    // Idiff(:,:,:,ind_S0) = [];
                    for (int i = 0; i < ind_S0.n_elem; ++i)
                        Vdiff[ind_S0(i)].reset();

                    for (sword graddir = Ngrad - 1; graddir >=0; --graddir)
                        if (Vdiff[graddir].n_elem == 0)  Vdiff.erase(Vdiff.begin() + graddir);

                    int pSize = Vdiff.size();

                    //Idiff = cat(4,S0_est,Idiff);
                    Vdiff.resize( pSize + ind_S0.n_elem);
                    for (sword i = pSize -1 ; i >= 0; --i)
                        Vdiff[i + ind_S0.n_elem] = Vdiff[i];

                    for (uword i = 0; i < ind_S0.n_elem; ++i)
                        Vdiff[i] = S0_est[i];

                    // normalize the signal S/S0
                    // for graddir = 1:Ngrad_mod
#pragma omp parallel for
                    for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                        //Idiff(:,:,:,graddir) = squeeze(Idiff(:,:,:,graddir).* Imask)./(S0_est + eps);
                        Vdiff[graddir] =  Vdiff[graddir] / (S0_est_M + std::numeric_limits<double>::epsilon());
                    }

                    // Repair the signal
                    // Idiff(Idiff>1) = 1;
                    // Idiff(Idiff<0) = 0;
#pragma omp parallel for
                    for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                        Vdiff[graddir].elem( find(Vdiff[graddir] > 1.0) ).ones();
                        Vdiff[graddir].elem( find(Vdiff[graddir] < 0.0) ).zeros();
                    }
                    //inda = find(squeeze(Idiff(:,:,1))>0);
                    uvec inda_vec = find( Vmask != 0);
                    Mat<uword> inda = conv_to<Mat<uword>>::from(inda_vec);

                    //totalNvoxels = prod(size(Imask));
                    size_t  totalNvoxels = Vmask.n_elem;

                    //allIndexes = repmat(inda(:)',[Ngrad 1]);
                    Mat<uword> allIndixes = repmat(inda.t(),Ngrad,1);
                    //allIndixes.save("inda.txt",arma::raw_ascii);

                    // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
                    Mat<uword> ind = allIndixes + (totalNvoxels *  repmat(linspace<Mat<uword>>(0, Ngrad-1,Ngrad),1,inda.n_elem));
                    Mat<T> diffSignal(Ngrad,inda.n_elem);

#pragma omp parallel for
                    for (uword j = 0; j < inda.n_elem; ++j)
                        for (uword i = 0; i < Ngrad; ++i)
                            diffSignal.at(i,j) = Vdiff[i].at( inda_vec.at(j)  );  //Vdiff[i].at(ind(i,j));

                    Mat<T> ODF;
                    switch (opts.reconsMethod) {
                        case QBI_DOTR2:
                        {
                            uvec indb0, indb1;
                            //indb0 = find(sum(diffGrads,2) == 0);
                            indb0 = find(sum(diffGrads, 1) == 0);

                            //indb1 = find(sum(diffGrads,2) ~= 0);
                            indb1 = find(sum(diffGrads, 1) != 0);

                            //tempSignal = diffSignal(indb1,:);
                            Mat<T> tempSignal = diffSignal.rows(indb1);
                            Mat<T> tempS0;
                            //if length(indb0)>1
                            if (indb0.n_elem > 1)
                                //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                //else
                            else
                                //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);

                            // tempVar = -log(tempSignal./tempS0)./repmat(diffBvals(indb1),[1 size(tempSignal,2)]);
                            Mat<T> tempVar = -log(tempSignal / tempS0) / repmat(diffBvals.rows(indb1), 1, size(tempSignal, 1));

                            // ADC = abs(tempVar);
                            Mat<T> ADC = abs(tempVar);

                            //Dt_nonoise = ADC*opts.qbi_dotr2.t;
                            Mat<T> Dt_nonoise = ADC * opts.dotr2.t;

                            //Signal_profile = opts.qbi_dotr2.eulerGamma + log(1./Dt_nonoise);
                            Mat<T> Signal_profile = opts.dotr2.eulerGamma + log(1.0 / Dt_nonoise);

                            //coeff0 = Kernel*Signal_profile;
                            Mat<T> coeff0 = Kernel * Signal_profile;

                            // coeff0 = coeff0./repmat(coeff0(1,:),[size(coeff0,1) 1]);
                            coeff0 = coeff0 / repmat(coeff0.row(0), size(coeff0, 0), 1);

                            //K_dot_r2(1) = (4/pi)*(2*sqrt(pi))/16;
                            K_dot_r2(0) = (4 / datum::pi) * (2 * std::sqrt(datum::pi)) / 16.0;

                            //ss = coeff0.*repmat(K_dot_r2,[1 size(coeff0,2)]);
                            Mat<T> ss = coeff0 % repmat(K_dot_r2, 1, size(coeff0, 1));

                            // ODF = basisV*ss;
                            ODF = basisV * ss;

                            // ODF = ODF - repmat(min(ODF),size(V,1),1);
                            ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                            // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                            ODF = ODF / repmat(sum(ODF,0), size(V, 0), 1);
                        }
                            break;
                        case QBI_CSA:
                        {
                            uvec indb0, indb1;
                            //indb0 = find(sum(diffGrads,2) == 0);
                            indb0 = find(sum(diffGrads, 1)==0);

                            //indb1 = find(sum(diffGrads,2) ~= 0);
                            indb1 = find(sum(diffGrads, 1)!=0);

                            //tempSignal = diffSignal(indb1,:);
                            Mat<T> tempSignal = diffSignal.rows(indb1) ;
                            Mat<T> tempS0;
                            //if length(indb0)>1
                            if (indb0.n_elem > 1)
                                //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                //else
                            else
                                //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);

                            // Signal_profile = log(-log(tempSignal./tempS0));
                            Mat <T> Signal_profile = log(-log(tempSignal / tempS0));

                            // coeff0 = Kernel*Signal_profile;
                            Mat<T> coeff0 = Kernel * Signal_profile;

                            //coeff0(1,:) = ones(1,size(coeff0,2));
                            coeff0.row(0) = ones<Row<T>>(size(coeff0, 1));

                            // ss = coeff0.*repmat(K_csa,[1 size(coeff0,2)]);
                            Mat<T> ss = coeff0 % repmat (K_csa ,1 , size(coeff0, 1));

                            // ODF = basisV*ss;
                            ODF = basisV * ss;

                            // ODF = ODF - repmat(min(ODF),size(V,1),1);
                            ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                            // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                            ODF = ODF/repmat(sum(ODF,0), size(V, 0), 1);
                        }
                            break;
                        case DSI:
                            break;
                        case GQI_L1:
                        {
                            uvec indb0, indb1;
                            //indb0 = find(sum(diffGrads,2) == 0);
                            indb0 = find(sum(diffGrads, 1)==0);

                            //indb1 = find(sum(diffGrads,2) ~= 0);
                            indb1 = find(sum(diffGrads, 1)!=0);

                            //tempSignal = diffSignal(indb1,:);
                            Mat<T> tempSignal = diffSignal.rows(indb1) ;
                            Mat<T> tempS0;
                            //if length(indb0)>1
                            if (indb0.n_elem > 1)
                                //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                //else
                            else
                                //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);
                            //end
                            //tempSignal = tempSignal./tempS0;
                            tempSignal = tempSignal / tempS0;

                            //ODF = Kernel*tempSignal;
                            ODF = Kernel * tempSignal;

                            //ODF = ODF - repmat(min(ODF),size(V,1),1);
                            ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);

                            //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                            ODF = ODF/repmat(sum(ODF, 0), size(V, 0), 1);
                        }
                            break;
                        case GQI_L2:
                        {
                            uvec indb0, indb1;
                            //indb0 = find(sum(diffGrads,2) == 0);
                            indb0 = find(sum(diffGrads, 1)==0);

                            //indb1 = find(sum(diffGrads,2) ~= 0);
                            indb1 = find(sum(diffGrads, 1)!=0);

                            //tempSignal = diffSignal(indb1,:);
                            Mat<T> tempSignal = diffSignal.rows(indb1) ;
                            Mat<T> tempS0;
                            //if length(indb0)>1
                            if (indb0.n_elem > 1)
                                //tempS0 = repmat(mean(diffSignal(indb0,:)),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(mean(diffSignal.rows(indb0)), indb1.n_elem, 1);
                                //else
                            else
                                //tempS0 = repmat(diffSignal(indb0,:),[length(indb1) 1]); % Signal from B0
                                tempS0 = repmat(diffSignal.rows(indb0), indb1.n_elem, 1);
                            //end
                            //tempSignal = tempSignal./tempS0;
                            tempSignal = tempSignal / tempS0;

                            //ODF = Kernel*tempSignal*opts.gqi_l2.lambda^3/pi;
                            ODF = Kernel * tempSignal * pow(opts.gqi.lambda, 3) / datum::pi;

                            //ODF = ODF - repmat(min(ODF),size(V,1),1);
                            ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                            //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                            ODF = ODF/repmat(sum(ODF,0), size(V, 0), 1);
                        }
                            break;
                        case QBI:
                        {
                            uvec indb0, indb1;
                            Mat<T> coeff, ss;

                            // indb0 = find(sum(diffGrads,2) == 0);
                            indb0 = find(sum(diffGrads, 1)==0);

                            // indb1 = find(sum(diffGrads,2) ~= 0);
                            indb1 = find(sum(diffGrads, 1)!=0);

                            // coeff = Kernel*diffSignal(indb1,:);
                            coeff = Kernel * diffSignal.rows(indb1);

                            // ss = coeff.*repmat(K,[1 size(coeff,2)]);
                            ss = coeff % repmat(K, 1, size(coeff,1));

                            // ODF = basisV*ss;
                            ODF = basisV * ss;

                            // ODF = ODF - repmat(min(ODF),size(V,1),1);
                            ODF = ODF - repmat(min(ODF,0), size(V, 0), 1);

                            // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                            ODF = ODF/repmat(sum(ODF,0), size(V, 0), 1);
                        }
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
                            for (uword i = 0; i < inda.n_elem; ++i) {
                                slicevf_CSF.at(inda.at(i)) = ODF.at(ODF.n_rows - 2 ,i);
                            }

#pragma omp parallel for
                            for (uword i = 0; i < inda.n_elem; ++i) {
                                slicevf_GM.at(inda.at(i)) = ODF.at(ODF.n_rows - 1 ,i);
                            }

                            //ODF_iso = ODF(end,:) + ODF(end-1,:);
                            ODF_iso = ODF.row(ODF.n_rows - 1) + ODF.row(ODF.n_rows - 2);

                            // ODF = ODF(1:end-2,:);
                            ODF.resize(ODF.n_rows - 2, ODF.n_cols);

                            //volvf_WM(inda) = sum(ODF,1);    % Volume fraction of WM
                            Row<T> tsum = sum(ODF,0);
#pragma omp parallel for
                            for (uword i = 0; i < inda.n_elem; ++i) {
                                slicevf_WM.at(inda.at(i)) = tsum.at(i);
                            }

                            // Adding the isotropic components to the ODF
                            ODF = ODF + repmat( ODF_iso / ODF.n_rows , ODF.n_rows, 1 );
                            ODF = ODF / repmat( sum(ODF,0) + std::numeric_limits<double>::epsilon() , ODF.n_rows,  1 );

                            // std(ODF,0,1)./( sqrt(mean(ODF.^2,1)) + eps )
                            Row<T>  temp = stddev( ODF, 0, 0 ) / sqrt(mean(pow(ODF,2),0) + std::numeric_limits<double>::epsilon());
#pragma omp parallel for
                            for (uword i = 0; i < inda.n_elem; ++i) {
                                slicevf_GFA.at(inda.at(i)) = temp.at(i);
                            }
                            break;
                    }
                    // % Allocating memory to save the ODF
                    // globODFslice = zeros(xdiff,ydiff,Nd);

                    // % Reordering ODF
                    //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                    Mat<uword> allIndexesODF = repmat(inda.t(),ODF.n_rows,1);

                    // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                    Mat<uword> ODFindexes = allIndexesODF + totalNvoxels * repmat(linspace<Mat<uword>>(0, ODF.n_rows - 1, ODF.n_rows ),1,inda.n_elem);

                    // globODFslice(ODFindexes(:)) = ODF(:);
                    uword slice_size = xdiff * ydiff * zdiff;
#pragma omp parallel for
                    for (uword j = 0; j < ODF.n_cols; ++j) {
                        for (uword i = 0; i < ODF.n_rows; ++i) {
                            globODFslice[i].at(ODFindexes(i,j) % slice_size) = ODF.at(i ,j);
                        }
                    }
                }
                if (opts.reconsMethod == RUMBA_SD) {
#pragma omp parallel for
                    for (auto i = 0; i < xdiff; ++i) {
                        Index3DType coord;
                        coord[0] = i;
                        for (auto j = 0; j < ydiff; ++j) {
                            coord[1] = j;
                            for (auto k = 0; k < zdiff; ++k) {
                                T temp = slicevf_CSF.at(i,j,k);
                                coord[2] = k;
                                imageCSF->SetPixel(coord, temp);
                            }
                        }
                    }

#pragma omp parallel for
                    for (auto i = 0; i < xdiff; ++i) {
                        Index3DType coord;
                        coord[0] = i;
                        for (auto j = 0; j < ydiff; ++j) {
                            coord[1] = j;
                            for (auto k = 0; k < zdiff; ++k) {
                                T temp = slicevf_GM.at(i,j,k);
                                coord[2] = k;
                                imageGM->SetPixel(coord, temp);
                            }
                        }
                    }

#pragma omp parallel for
                    for (auto i = 0; i < xdiff; ++i) {
                        Index3DType coord;
                        coord[0] = i;
                        for (auto j = 0; j < ydiff; ++j) {
                            coord[1] = j;
                            for (auto k = 0; k < zdiff; ++k) {
                                T temp = slicevf_WM.at(i,j,k);
                                coord[2] = k;
                                imageWM->SetPixel(coord, temp);
                            }
                        }
                    }

#pragma omp parallel for
                    for (auto i = 0; i < xdiff; ++i) {
                        Index3DType coord;
                        coord[0] = i;
                        for (auto j = 0; j < ydiff; ++j) {
                            coord[1] = j;
                            for (auto k = 0; k < zdiff; ++k) {
                                T temp = slicevf_GFA.at(i,j,k);
                                coord[2] = k;
                                imageGFA->SetPixel(coord, temp);
                            }
                        }
                    }
                }
#pragma omp parallel for
                for (auto n = 0; n < Nd; ++n) {
                    Cube<T> tempc = globODFslice[n];
                    Index4DType coord;
                    coord[3] = n;
                    for (auto i = 0; i < xdiff; ++i) {
                        coord[0] = i;
                        for (auto j = 0; j < ydiff; ++j) {
                            coord[1] = j;
                            for (auto k = 0; k < zdiff; ++k) {
                                T temp = tempc.at(i,j,k);
                                coord[2] = k;
                                imageODF->SetPixel(coord, temp);
                            }
                        }
                    }
                }
            }
                break;
        }

        if (opts.reconsMethod == RUMBA_SD) {
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
        }

        LOG_INFO << "writting file " << ODFfilename;
        WriteImage<Image4DType,NiftiType>(ODFfilename,imageODF);
    }
}

#endif

