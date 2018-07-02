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
#include "create_kernel_for_dti.hpp"
#include "constants.hpp"
#include "mirt3D.hpp"
#include "362_shell_semisphere.hpp"

#include "eigenvaluefield33.hpp"
#include "eigenvectorfield33.hpp"
#include "lsqnonnegvect.hpp"

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
                                         ) {
    using namespace arma;
    using namespace itk;

    Mat<T> V;

    if (opts.ODFDirscheme.length() > 0) {
        V.load(diffGradsfilename, arma::raw_ascii);
    } else {
        V = ODFDirscheme_362<T>();
    }
    Col<T> norma = sqrt(sum(pow(V,2),1));
    V = V/repmat(norma,1,3);
    int Nd = V.n_rows;


    // Creating replicated Direction Scheme
    Mat<T> Vall = arma::join_cols(V,-1*V);

    // Orders of the replicated Scheme
    ucolvec orderAll = linspace<ucolvec>(0, Nd - 1, Nd);
    orderAll = repmat(orderAll,2,1);


    // Creating Neighborhood Matrix for Peaks extraction
    int Np = 3; // Number of peaks per voxel
    float Beta = 15; // Neighborhood angle to look for the peaks
    float c = sqrt(2*(1-cos(Beta*datum::pi/180))); // Assuming sphere with radius = 1. Distance threshold


    // Saving the neighborhood for each scheme vertex
    field<ucolvec> neighMat(Nd,1);
    Col<T> dVect;
    Mat<T> tempVar;


    for (int i = 0; i < Nd; ++i) {

        // Euclidean distance between each point and the rest
        tempVar = Vall - repmat(Vall.row(i),2*Nd,1);
        dVect = sqrt(sum(pow(tempVar,2),1));

        uvec tempIndex = orderAll(find((dVect <= c)&&(dVect != 0))); // Vertices belonging to the desired neighborhood
        neighMat(i,0) = tempIndex; // Saving Neighborhoods
        //        LOG_INFO << neighMat(i,0);
    }



    Mat<T> diffGrads;
    diffGrads.load(diffGradsfilename, arma::raw_ascii);
    diffGrads = diffGrads.t();

    LOG_INFO << "Reading diffGrads " << diffGradsfilename << " [" << diffGrads.n_rows << ", " << diffGrads.n_cols
             << "]";

    size_t Ngrad = diffGrads.n_rows;

    //  vector normalization: obtaining unit vectors (0-vectors are preserved)
    //norm_factor = sqrt(sum(diffGrads.^2,2));
    Col<T> norm_factor = sqrt(sum(pow(diffGrads, 2), 1)) + std::numeric_limits<double>::epsilon();
    // diffGrads = diffGrads./repmat(norm_factor + eps,[1 3]);
    diffGrads = diffGrads / repmat(norm_factor, 1, 3);

    Mat<T> diffBvalsM;
    diffBvalsM.load(diffBvalsfilename, arma::raw_ascii);

    Col<T> diffBvals(diffBvalsM.n_elem);
    for (size_t i = 0; i < diffBvalsM.n_elem; ++i)
        diffBvals(i) = diffBvalsM(0, i);

    LOG_INFO << "Reading diffBvals " << diffBvalsfilename << " [" << diffBvals.n_elem << "]";
    LOG_INFO << "The measurement scheme contains " << Ngrad << " volumes";
    LOG_INFO << "The max b-value is " << diffBvals.max();

    // --- volumes where b-value = 0
    // indb0 = find(diffBvals.*sum(diffGrads.^2,2) == 0);
    //        uvec ind_S0_vec = find(diffBvals % sum(pow(diffGrads, 2), 1) <= 10);
    uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
    uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);

    //    Mat<T> indb0 = conv_to<Mat<T>>::from(ind_S0_vec);
    //    Mat<T> indb1 = conv_to<Mat<T>>::from(indb1_vec);

    // display(['From the ' num2str(Ngrad) ' volumes, ' num2str(length(indb0)) ' are b0 images']);
    LOG_INFO << "From the " << Ngrad << " volumes, " << indb0.n_elem << " are b0 images";

    // in rumba_sd the data will be modified/reordered
    //diffGrads(indb0,:) = [];
    //diffGrads = [0 0 0; diffGrads];
    //diffBvals(indb0) = [];
    //diffBvals = [0; diffBvals];
    //size_t Ngrad_mod = diffGrads.n_rows;

    Image4DType::Pointer imageDiff = Image4DType::New();
    ReadImage<Image4DType>(diffSignalfilename, imageDiff);

    Image4DType::RegionType regionDiff = imageDiff->GetBufferedRegion();
    Index4DType index4D = regionDiff.GetIndex();
    Size4DType size4D = regionDiff.GetSize();
    Spacing4DType spacing4D = imageDiff->GetSpacing();
    Origin4DType origin4D = imageDiff->GetOrigin();
    Direction4DType direction4D = imageDiff->GetDirection();

    // xdiff = Vdiff(1).dim(1);
    // ydiff = Vdiff(1).dim(2);
    // zdiff = Vdiff(1).dim(3);
    int xdiff = regionDiff.GetSize()[0];
    int ydiff = regionDiff.GetSize()[1];
    int zdiff = regionDiff.GetSize()[2];

    std::vector<Cube<T>> Vdiff(regionDiff.GetSize()[3]);

#pragma omp parallel for
    for (int w = 0; w < regionDiff.GetSize()[3]; ++w) {
        Cube<T> temp(regionDiff.GetSize()[0], regionDiff.GetSize()[1], regionDiff.GetSize()[2]);
        Index4DType coord_4d;
        coord_4d[3] = w;
        for (int x = 0; x < regionDiff.GetSize()[0]; ++x) {
            coord_4d[0] = x;
            for (int y = 0; y < regionDiff.GetSize()[1]; ++y) {
                coord_4d[1] = y;
                for (int z = 0; z < regionDiff.GetSize()[2]; ++z) {
                    coord_4d[2] = z;
                    temp.at(x, y, z) = imageDiff->GetPixel(coord_4d);
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
    Index3DType index3D = regionMask.GetIndex();
    Size3DType size3D = regionMask.GetSize();
    Spacing3DType spacing3D = imageMask->GetSpacing();
    Origin3DType origin3D = imageMask->GetOrigin();
    Direction3DType direction3D = imageMask->GetDirection();

    LOG_INFO << "Mask size " << regionMask.GetSize();

    Cube<T> Vmask(regionMask.GetSize()[0], regionMask.GetSize()[1], regionMask.GetSize()[2]);

#pragma omp parallel for
    for (int x = 0; x < regionMask.GetSize()[0]; ++x) {
        Index3DType coord_3d;
        coord_3d[0] = x;
        for (int y = 0; y < regionMask.GetSize()[1]; ++y) {
            coord_3d[1] = y;
            for (int z = 0; z < regionMask.GetSize()[2]; ++z) {
                coord_3d[2] = z;
                Vmask.at(x, y, z) = imageMask->GetPixel(coord_3d);
            }
        }
    }

    // %%  ================== Creating ODF Volume ============================== %

    Image4DType::Pointer imageODF = Image4DType::New();
    Size4DType size4DODF;
    Index4DType index4DODF;

    size4DODF[0] = xdiff;
    index4DODF[0] = 0;
    size4DODF[1] = ydiff;
    index4DODF[1] = 0;
    size4DODF[2] = zdiff;
    index4DODF[2] = 0;
    size4DODF[3] = Nd;
    index4DODF[3] = 0;

    CreateImage<Image4DType>(imageODF, size4DODF, index4DODF, spacing4D, origin4D, direction4D);
    LOG_INFO << "created ODF image";
    // LOG_INFO << imageODF;

    // %%  =====================================================================

    Mat<T> Kernel(V.n_rows + 2, diffGrads.n_rows);
    Mat<T> basisV;
    Col<T> K;
    Col<T> K_dot_r2;
    Col<T> K_csa;
    Col<T> thetaV;
    Col<T> phiV;
    Mat<uword> qspace;
    Mat<T> xi;
    Mat<T> yi;
    Mat<T> zi;
    Mat<T> rmatrix;
    Mat<T> C;
    uword Lmax;
    uword Nmin;
    Kernel.zeros();

    obtain_Lmax(diffGrads, Lmax, Nmin);

    if (Lmax >= 8)
    {
        Lmax = 8;
    }
    switch (opts.reconsMethod) {
    case RUMBA_SD:
        create_Kernel_for_rumba<T>(V, diffGrads, diffBvals, Kernel, opts);

        //      Standard Spherical Harmonics Basis
        //        construct_SH_basis<T>(Lmax, V, 2, "real", thetaV, phiV, basisV);

        //      MRTRIX 3.0 Spherical Harmonics Basis
        if (opts.save_mrtrix)
            construct_SH_basis_MrTrix<T>(Lmax, V, 2, "complex", thetaV, phiV, basisV);
        else
            construct_SH_basis<T>(Lmax, V, 2, "real", thetaV, phiV, basisV);

        LOG_INFO << "Created kernel for RUMBA";
        break;
    case QBI_DOTR2:
    case QBI_CSA:
        if (opts.save_mrtrix)   // Standard Spherical Harmonics Basis
            create_Kernel_for_dotr2_MrTrix<T>(V, diffGrads, diffBvals, Kernel, basisV, K_dot_r2, K_csa, opts);
        else  // MRTRIX 3.0 Spherical Harmonics Basis
            create_Kernel_for_dotr2<T>(V, diffGrads, diffBvals, Kernel, basisV, K_dot_r2, K_csa, opts);

        LOG_INFO << "Created kernel for CSA/DOTR2";
        break;
    case DSI:

        //      Standard Spherical Harmonics Basis
        create_Kernel_for_dsi<T>(V, diffGrads, diffBvals, Kernel, basisV, qspace, xi, yi, zi, rmatrix, opts);

        //      MRTRIX 3.0 Spherical Harmonics Basis
        if (opts.save_mrtrix)
            create_Kernel_for_dsi_MrTrix<T>(V, diffGrads, diffBvals, Kernel, basisV, qspace, xi, yi, zi, rmatrix, opts);

        LOG_INFO << "Created kernel for DSI";
        break;
    case QBI:
        if (opts.save_mrtrix)   // Standard Spherical Harmonics Basis
            create_Kernel_for_qbi_MrTrix<T>(V, diffGrads, diffBvals, Kernel, basisV, K, opts);
        else  // MRTRIX 3.0 Spherical Harmonics Basis
            create_Kernel_for_qbi<T>(V, diffGrads, diffBvals, Kernel, basisV, K, opts);

        LOG_INFO << "Created kernel for QBI";
        break;
    case GQI_L1:
    case GQI_L2:
        create_Kernel_for_gqi<T>(V, diffGrads, diffBvals, Kernel, opts);

        //      MRTRIX 3.0 Spherical Harmonics Basis
        if (opts.save_mrtrix)
            construct_SH_basis_MrTrix<T>(Lmax, V, 2, "complex", thetaV, phiV, basisV);

        LOG_INFO << "Created kernel for GQI_L1/GQI_L2";
        break;
    case DTI_NNLS:
        create_Kernel_for_dti<T>(V, diffGrads, diffBvals, Kernel, C, opts);
        LOG_INFO << "Created kernel for DTI_NNLS";
        break;
    }



    // %%  ================== Creating SH Volume ============================== %
    int Nsh = basisV.n_cols;
    //        LOG_INFO << Nsh;
    Image4DType::Pointer imageSH = Image4DType::New();
    Image4DType::Pointer imageSH_WM = Image4DType::New();
    Size4DType size4DSH;
    Index4DType index4DSH;

    size4DSH[0] = xdiff;
    index4DSH[0] = 0;
    size4DSH[1] = ydiff;
    index4DSH[1] = 0;
    size4DSH[2] = zdiff;
    index4DSH[2] = 0;
    size4DSH[3] = Nsh;
    index4DSH[3] = 0;

    CreateImage<Image4DType>(imageSH, size4DSH, index4DSH, spacing4D, origin4D, direction4D);
    CreateImage<Image4DType>(imageSH_WM, size4DSH, index4DSH, spacing4D, origin4D, direction4D);
    LOG_INFO << "created SH image";
    // LOG_INFO << imageSH;

    // %%  =====================================================================



    // %%  ================== Creating Peaks Volume ============================== %
    Image4DType::Pointer imagePeaks = Image4DType::New();
    Size4DType size4DPeaks;
    Index4DType index4DPeaks;

    size4DPeaks[0] = xdiff;
    index4DPeaks[0] = 0;
    size4DPeaks[1] = ydiff;
    index4DPeaks[1] = 0;
    size4DPeaks[2] = zdiff;
    index4DPeaks[2] = 0;
    size4DPeaks[3] = Np*3;
    index4DPeaks[3] = 0;

    CreateImage<Image4DType>(imagePeaks, size4DPeaks, index4DPeaks, spacing4D, origin4D, direction4D);
    LOG_INFO << "created Peaks image";

    // %%  =====================================================================


    // %%  ================== Creating Filenames  ============================== %

    std::string filenameCSF;
    std::string filenameGM;
    std::string filenameWM;
    std::string filenameGFA;
    std::string filenameSH;
    std::string filenameSH_WM;
    std::string filenamePeaks;

    if (opts.zip) {
        filenameCSF = opts.outputDir + "_vfcsf.nii.gz";
        filenameGM = opts.outputDir + "_vfgm.nii.gz";
        filenameWM = opts.outputDir + "_vfwm.nii.gz";
        filenameGFA = opts.outputDir + "_gfa.nii.gz";
        filenameSH_WM = opts.outputDir + "_sh_wm.nii.gz";
        filenameSH = opts.outputDir + "_sh.nii.gz";
        filenamePeaks = opts.outputDir + "_peaks.nii.gz";

    } else {
        filenameCSF = opts.outputDir + "_vfcsf.nii";
        filenameGM = opts.outputDir + "_vfgm.nii";
        filenameWM = opts.outputDir + "_vfwm.nii";
        filenameGFA = opts.outputDir + "_gfa.nii";
        filenameSH_WM = opts.outputDir + "_sh_wm.nii";
        filenameSH = opts.outputDir + "_sh.nii";
        filenamePeaks = opts.outputDir + "_peaks.nii";

    }
    // %%  ================== Creating Filenames  ============================== %

    // %%  ================== Creating 3D Volumes  ============================== %

    size3D[0] = xdiff;
    index3D[0] = 0;
    size3D[1] = ydiff;
    index3D[1] = 0;
    size3D[2] = zdiff;
    index3D[2] = 0;

    Image3DType::Pointer imageCSF = Image3DType::New();
    CreateImage<Image3DType>(imageCSF, size3D, index3D, spacing3D, origin3D, direction3D);

    Image3DType::Pointer imageGM = Image3DType::New();
    CreateImage<Image3DType>(imageGM, size3D, index3D, spacing3D, origin3D, direction3D);

    Image3DType::Pointer imageWM = Image3DType::New();
    CreateImage<Image3DType>(imageWM, size3D, index3D, spacing3D, origin3D, direction3D);

    Image3DType::Pointer imageGFA = Image3DType::New();
    CreateImage<Image3DType>(imageGFA, size3D, index3D, spacing3D, origin3D, direction3D);

    // %%  ============================================================================ %

    switch (opts.datreadMethod) {
    case VOXELS: {
        Mat<T> ODF;
        Mat<T> ss;
        Mat<T> slice_GFA(xdiff, ydiff, fill::zeros);

        Cube<T> globODFslice(xdiff, ydiff, Nd, fill::zeros);
        Cube<T> globSHslice(xdiff, ydiff, Nsh, fill::zeros);

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
                    Idiff.slice(graddir) =
                            Vdiff[graddir].slice(slice) % Vmask.slice(slice); // product-wise multiplication
                }

                // Voxel indexes
                // inda = find(squeeze(Idiff(:,:,1))>0);
                uvec inda_vec = find(Idiff.slice(0) > 0);
                Mat<uword> inda = conv_to<Mat<uword>>::from(inda_vec);

                // totalNvoxels = prod(size(Imask)); % Total Number of voxels in the slice
                uword totalNvoxels = Imask.n_rows * Imask.n_cols;

                // allIndexes = repmat(inda(:)',[Ngrad 1]); % Indexes
                Mat<uword> allIndexes = repmat(inda.t(), Ngrad, 1);

                // Diffusion signal matrix evaluated in the indexes inside brain mask
                // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
                Mat<T> diffSignal(Ngrad, inda.n_elem);
                Mat<uword> ind = allIndexes + (totalNvoxels *
                                               repmat(linspace<Mat<uword>>(0, Ngrad - 1, Ngrad), 1,
                                                      inda.n_elem));
#pragma omp parallel for
                for (uword j = 0; j < inda.n_elem; ++j)
                    for (uword i = 0; i < Ngrad; ++i)
                        diffSignal.at(i, j) = Idiff.at(ind(i, j));


                ODF.resize(rmatrix.n_rows, inda.n_elem);
                switch (opts.reconsMethod) {
                case RUMBA_SD:
                case QBI:
                case GQI_L1:
                case GQI_L2:
                case QBI_DOTR2:
                case QBI_CSA:
                case DTI_NNLS:
                    break;
                case DSI: {
                    // --- Signal in the 3D image
                    // Smatrix = SignalMatrixBuilding(qspace,diffSignal,opts.dsi.resolution)
                    Mat<T> tempSignal;

                    //                                uvec indb0, indb1;
                    //                                // indb0 = find(sum(diffGrads,2) == 0);
                    //                               indb0 = find(sum(diffGrads, 1) == 0);

                    //                               // indb1 = find(sum(diffGrads,2) ~= 0);
                    //                               indb1 = find(sum(diffGrads, 1) != 0);

                    //                    uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
                    //                    uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);


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
                    //#pragma omp parallel for
                    for (uword indvox = 0; indvox < inda.n_elem; ++indvox) {

                        // Smatrix = SignalMatrixBuilding_Volume(qspace,tempSignal(:,indvox),opts.dsi.resolution);
                        Cube<T> Smatrix = SignalMatrixBuilding_Volume(qspace, conv_to<Col<T>>::from(
                                                                          tempSignal.col(indvox)), opts.dsi.resolution);

                        af::array Smatrix_af = af::array(Smatrix.n_rows, Smatrix.n_cols, Smatrix.n_slices,
                                                         Smatrix.memptr());

                        // --- DSI: PDF computation via fft
                        // Pdsi = real(fftshift(fftn(ifftshift(Smatrix))));
                        af::array Pdsi_af = af::real(fftshift3(fft3(ifftshift3(Smatrix_af))));

                        Cube<T> Pdsi(size(Smatrix));
                        Pdsi_af.host(Pdsi.memptr());

                        // Pdsi(Pdsi<0)=0;
                        Pdsi.elem(find(Pdsi < 0.0)).zeros();

                        // Pdsi_int = mirt3D_mexinterp(Pdsi,xi,yi,zi);

                        Mat<T> Pdsi_int = mirt3D_Function(Pdsi, xi, yi, zi);

                        // Pdsi_int(Pdsi_int<0) = 0;
                        Pdsi_int.elem(find(Pdsi_int < 0.0)).zeros();

                        // Pdsi_int = Pdsi_int./sum(Pdsi_int(:));
                        Pdsi_int = Pdsi_int / accu(Pdsi_int);

                        // --- Numerical ODF-DSI reconstruction

                        // ODFvox = sum(Pdsi_int.*(rmatrix.^2),2);
                        Col<T> ODFvox = sum(Pdsi_int % pow(rmatrix, 2), 1);

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
                    ODF = abs(basisV * sphE_dsi);
                    ODF = ODF - min(ODF);
                    if (opts.norm_odfs)    {
                        // ODF = ODF./repmat(sum(ODF),[size(ODF,1) 1]);

                        ODF = ODF / repmat(sum(ODF), size(ODF, 0), 1);
                    }
                    Mat<T> TempMat = trans(basisV)*basisV;
                    ss = inv(TempMat)*trans(basisV)*ODF;

                    ss =ss/repmat(sum(abs(ss))+ std::numeric_limits<double>::epsilon(),Nsh,1);

                }
                    break;
                }

                // Allocating memory to save the ODF
                // globODFslice = zeros(xdiff,ydiff,Nd);

                // % Reordering ODF
                //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                Mat<uword> allIndexesODF = repmat(inda.t(), ODF.n_rows, 1);

                // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                Mat<uword> ODFindexes = allIndexesODF + totalNvoxels *
                        repmat(linspace<Mat<uword>>(0, ODF.n_rows - 1, ODF.n_rows), 1, inda.n_elem);

                // % Reordering SH
                //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                Mat<uword> allIndexesSH = repmat(inda.t(), Nsh, 1);

                // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                Mat<uword> SHindexes = allIndexesSH + totalNvoxels * repmat(linspace<uvec>(0, Nsh - 1, Nsh), 1, inda.n_elem);


                if (opts.reconsMethod != DTI_NNLS){
                    Row<T> temp = stddev(ODF, 0, 0) /
                            sqrt(mean(pow(ODF, 2), 0) + std::numeric_limits<double>::epsilon());

#pragma omp parallel for
                    for (uword i = 0; i < inda.n_elem; ++i) {
                        slice_GFA.at(inda.at(i)) = temp.at(i);
                    }
                }

                if (opts.reconsMethod != DTI_NNLS) {
                    // globODFslice(ODFindexes(:)) = ODF(:);

                    if (opts.save_mrtrix) {
#pragma omp parallel for
                        for (uword j = 0; j < ss.n_cols; ++j) {
                            for (uword i = 0; i < ss.n_rows; ++i) {
                                globSHslice.at(SHindexes.at(i, j)) = ss.at(i, j);
                            }
                        }
                    } else {
#pragma omp parallel for
                        for (uword j = 0; j < ODF.n_cols; ++j) {
                            for (uword i = 0; i < ODF.n_rows; ++i) {
                                globODFslice.at(ODFindexes(i, j)) = ODF.at(i, j);
                            }
                        }
                    }
                }

            }
            if (opts.reconsMethod != DTI_NNLS) {
#pragma omp parallel for
                for (uword i = 0; i < xdiff; ++i) {
                    Index4DType coord;
                    coord[0] = i;
                    coord[2] = slice;
                    for (uword j = 0; j < ydiff; ++j) {
                        coord[1] = j;
                        if (opts.save_mrtrix) {
                            for (uword k = 0; k < Nsh; ++k) {
                                coord[3] = k;
                                imageSH->SetPixel(coord, globSHslice.at(i, j, k));
                            }
                        } else {
                            for (uword k = 0; k < Nd; ++k) {
                                coord[3] = k;
                                imageODF->SetPixel(coord, globODFslice.at(i, j, k));
                            }
                        }
                    }
                }
            }


#pragma omp parallel for
            for (uword j = 0; j < ydiff; ++j) {
                Index3DType coord;
                coord[1] = j;
                coord[2] = slice;
                for (uword i = 0; i < xdiff; ++i) {
                    coord[0] = i;
                    imageGFA->SetPixel(coord, slice_GFA.at(i, j));
                }
            }

        }

    }
        break;
    case SLICES: {
        Mat<T> slicevf_CSF(xdiff, ydiff, fill::zeros);
        Mat<T> slicevf_GM(xdiff, ydiff, fill::zeros);
        Mat<T> slicevf_WM(xdiff, ydiff, fill::zeros);
        Mat<T> slice_GFA(xdiff, ydiff, fill::zeros);
        Row<T> ODF_iso(ydiff, fill::zeros);
        Mat<T> sliceL1(xdiff, ydiff, fill::zeros);
        Mat<T> sliceL2(xdiff, ydiff, fill::zeros);
        Mat<T> sliceL3(xdiff, ydiff, fill::zeros);

        Cube<T> globODFslice(xdiff, ydiff, Nd, fill::zeros);
        Cube<T> globSHslice(xdiff, ydiff, Nsh, fill::zeros);
        Cube<T> globSHWMslice(xdiff, ydiff, Nsh, fill::zeros);
        Cube<T> globPeaksslice(xdiff, ydiff, Np*3, fill::zeros);


        for (uword slice = 0; slice < zdiff; ++slice) {
            LOG_INFO << "Processing slice number " << slice << " of " << zdiff;

            // Imask  = squeeze(spm_slice_vol(Vmask,spm_matrix([0 0 slice]),Vmask.dim(1:2),0));
            //mat Imask = Vmask.slice(slice);

            globODFslice.zeros();
            globSHslice.zeros();
            globSHWMslice.zeros();
            globPeaksslice.zeros();

            slicevf_CSF.zeros();
            slicevf_GM.zeros();
            slicevf_WM.zeros();
            slice_GFA.zeros();

            // if sum(Imask(:)) ~= 0
            Cube<T> Idiff(xdiff, ydiff, Ngrad);
            for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                // Idiff(:,:,graddir)  = spm_slice_vol(Vdiff(graddir),spm_matrix([0 0 slice]),Vdiff(graddir).dim(1:2),0).*logical(Imask);
                Idiff.slice(graddir) =
                        Vdiff[graddir].slice(slice) % Vmask.slice(slice); // product-wise multiplication
            }

            // isolating the b0 images
            // S0_est = squeeze( Idiff(:,:,indb0) );
            Cube<T> S0_est(xdiff, ydiff, indb0.n_elem);
            Mat<T> S0_est_M(xdiff, ydiff);

            for (uword i = 0; i < indb0.n_elem; ++i) {
                S0_est.slice(i) = Idiff.slice(indb0(i));
            }

            S0_est_M = mean(S0_est, 2);
            // if there are several b0 images in the data, we compute the mean value
            if (indb0.n_elem > 1) {
                //#pragma omp parallel for
                for (uword i = 0; i < indb0.n_elem; ++i)
                    S0_est.slice(i) = S0_est_M % Vmask.slice(slice);
            }

            //  normalize the signal S/S0
            //for graddir = 1:Ngrad_mod
            for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                //Idiff(:,:,graddir) = squeeze(Idiff(:,:,graddir))./(S0_est + eps);
                Idiff.slice(graddir) =
                        Idiff.slice(graddir) / (S0_est_M + std::numeric_limits<double>::epsilon());
            }
            // Repair the signal
            // Idiff(Idiff>1) = 1;
            // Idiff(Idiff<0) = 0;
            Idiff.elem(find(Idiff > 1.0)).ones();
            Idiff.elem(find(Idiff < 0.0)).zeros();

            //inda = find(squeeze(Idiff(:,:,1))>0);
            uvec inda_vec = find(Idiff.slice(0) > 0);
            Mat<uword> inda = conv_to<Mat<uword>>::from(inda_vec);

            //totalNvoxels = prod(size(Imask));
            uword totalNvoxels = Vmask.slice(slice).n_elem;

            Mat<T> ODF;
            Mat<T> ODF_WM;
            Mat<T> ss;
            Mat<T> ss_wm;
            Mat<T> ordpeaksMat;
            if (inda.n_elem > 0) {
                //allIndexes = repmat(inda(:)',[Ngrad 1]);
                Mat<uword> allIndexes = repmat(inda.t(), Ngrad, 1);
                //allIndixes.save("inda.txt",arma::raw_ascii);

                // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
                Mat<uword> ind = allIndexes + (totalNvoxels *
                                               repmat(linspace<Mat<uword>>(0, Ngrad - 1, Ngrad), 1,
                                                      inda.n_elem));
                Mat<T> diffSignal(Ngrad, inda.n_elem);

#pragma omp parallel for
                for (uword j = 0; j < inda.n_elem; ++j) {
                    for (uword i = 0; i < Ngrad; ++i)
                        diffSignal.at(i, j) = Idiff.at(ind.at(i, j));
                }
                switch (opts.reconsMethod) {
                case DTI_NNLS: {
                    Mat<T> x;

                    // x=lsqnonnegvect(Kernel,log(diffSignal./repmat(diffSignal(1,:),[size(diffSignal,1) 1])));
                    x = lsqnonnegvect<T>(Kernel,
                                         log(diffSignal / repmat(mean(diffSignal.rows(indb0)), diffSignal.n_rows, 1)));

                    // T = C' * x(1:size(V,1),:);
                    Mat<T> TT = C.t() * x.rows(regspace<uvec>(0, V.n_rows - 1));

                    // [l1,l2,l3] = eigenvaluefield33(T(6,:), T(5,:)/2, T(4,:)/2, T(3,:), T(2,:)/2, T(1,:));
                    Row<T> l1, l2, l3;
                    eigenvaluefield33<T>(TT.row(5), TT.row(4) / 2.0, TT.row(3) / 2.0, TT.row(2),
                                         TT.row(1) / 2.0, TT.row(0), l1, l2, l3);

                    // tempVar = sort([l1;l2;l3]);
                    Mat<T> tempVar = sort(join_vert(l1, join_vert(l2, l3)));

                    // l1 = tempVar(3,:);
                    l1 = tempVar.row(2);

                    // l2 = tempVar(2,:);
                    l2 = tempVar.row(1);

                    // l3 = tempVar(1,:);
                    l3 = tempVar.row(0);

                    // [e1x,e1y,e1z,e2x,e2y,e2z,e3x,e3y,e3z] = eigenvectorfield33( T(6,:), T(5,:)/2, T(4,:)/2, T(3,:), T(2,:)/2, T(1,:), l1, l2, l3);
                    std::vector<arma::Row<T>> e1;
                    std::vector<arma::Row<T>> e2;
                    std::vector<arma::Row<T>> e3;

                    eigenvectorfield33<T>(TT.row(5), TT.row(4) / 2.0, TT.row(3) / 2.0, TT.row(2),
                                          TT.row(1) / 2.0, TT.row(0), l1, l2, l3, e1, e2, e3);

                    //Tensor = [T(6,:); T(5,:)/2;T(4,:)/2; T(3,:); T(2,:)/2; T(1,:)];
                    Mat<T> Tensor;
                    Tensor.insert_rows(0, TT.row(5));
                    Tensor.insert_rows(1, TT.row(4) / 2.0);
                    Tensor.insert_rows(2, TT.row(3) / 2.0);
                    Tensor.insert_rows(3, TT.row(2));
                    Tensor.insert_rows(4, TT.row(1) / 2.0);
                    Tensor.insert_rows(5, TT.row(0));

                    // Nd = size(Tensor,1);
                    uword Ndt = Tensor.n_rows;

                    // slice2Save = zeros(xdiff,ydiff,Nd);
                    Cube<T> slice2Save = zeros<Cube<T>>(xdiff, ydiff, Ndt);

                    // allIndexesTensor = repmat(inda(:)',[Nd 1]); % Image indexes
                    umat allIndexesTensor = repmat(inda.t(), Ndt, 1);

                    // Tenindexes = allIndexesTensor + totalNvoxels*repmat([0:Nd-1]',[1 length(inda) ]); % Indexes in 4D
                    umat Tenindexes = allIndexesTensor +
                            totalNvoxels * repmat(regspace<uvec>(0, Ndt - 1), 1, inda.n_rows);

                    // slice2Save(Tenindexes(:)) = Tensor(:);


                    // sliceL1 = zeros(xdiff,ydiff);% Main eigenvalue
                    sliceL1.zeros();

                    // sliceL2 = zeros(xdiff,ydiff);% Second eigenvalue
                    sliceL2.zeros();

                    // sliceL3 = zeros(xdiff,ydiff);% Third eigenvalue
                    sliceL3.zeros();

                    // sliceL1(inda) = l1;
                    sliceL1(inda) = l1;

                    // sliceL2(inda) = l2;
                    sliceL2(inda) = l2;

                    // sliceL3(inda) = l3;
                    sliceL3(inda) = l3;

                    // sliceV1 = zeros(xdiff,ydiff,3);% Main eigenvector
                    Cube<T> sliceV1 = zeros<Cube<T>>(xdiff, ydiff, 3);

                    // tempSlice = zeros(xdiff,ydiff);
                    Mat<T> tempSlice = zeros<Mat<T>>(xdiff, ydiff);

                    // tempSlice(inda) = e1x;
                    tempSlice(inda) = e1[0];

                    // sliceV1(:,:,1) = tempSlice;
                    sliceV1.slice(0) = tempSlice;

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e1y;
                    tempSlice(inda) = e1[1];

                    // sliceV1(:,:,2) = tempSlice;
                    sliceV1.slice(1) = tempSlice;

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e1z;
                    tempSlice(inda) = e1[2];

                    // sliceV1(:,:,3) = tempSlice;
                    sliceV1.slice(2) = tempSlice;

                    // sliceV2 = zeros(xdiff,ydiff,3);% Second eigenvector
                    Cube<T> sliceV2 = zeros<Cube<T>>(xdiff, ydiff, 3);

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e2x;
                    tempSlice(inda) = e2[0];

                    // sliceV2(:,:,1) = tempSlice;
                    sliceV2.slice(0) = tempSlice;

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e2y;
                    tempSlice(inda) = e2[1];

                    // sliceV2(:,:,2) = tempSlice;
                    sliceV2.slice(1) = tempSlice;

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e2z;
                    tempSlice(inda) = e2[2];

                    // sliceV2(:,:,3) = tempSlice;
                    sliceV2.slice(2) = tempSlice;

                    // sliceV3 = zeros(xdiff,ydiff,3);% Third eigenvector
                    Cube<T> sliceV3 = zeros<Cube<T>>(xdiff, ydiff, 3);

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e3x;
                    tempSlice(inda) = e3[0];

                    // sliceV3(:,:,1) = tempSlice;
                    sliceV3.slice(0) = tempSlice;

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e3y;
                    tempSlice(inda) = e3[1];

                    // sliceV3(:,:,2) = tempSlice;
                    sliceV3.slice(1) = tempSlice;

                    // tempSlice = zeros(xdiff,ydiff);
                    tempSlice.zeros();

                    // tempSlice(inda) = e3z;
                    tempSlice(inda) = e3[2];

                    // sliceV3(:,:,3) = tempSlice;
                    sliceV3.slice(2) = tempSlice;
                }
                    break;
                case DSI: {


                }
                    break;
                case QBI_DOTR2: {
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

                    tempSignal = tempSignal / (tempS0+ std::numeric_limits<double>::epsilon());
                    tempSignal.elem(find(tempSignal > 1.0)).ones();
                    tempSignal.elem(find(tempSignal < 0.0)).zeros();

                    Mat<T> tempVar = -log(tempSignal + std::numeric_limits<double>::epsilon()) /
                            repmat(diffBvals.rows(indb1), 1, size(tempSignal, 1));

                    // ADC = abs(tempVar);
                    Mat<T> ADC = abs(tempVar);

                    //Dt_nonoise = ADC*opts.qbi_dotr2.t;
                    Mat<T> Dt_nonoise = ADC * opts.dotr2.t;

                    //Signal_profile = opts.qbi_dotr2.eulerGamma + log(1./Dt_nonoise);
                    Mat<T> Signal_profile = opts.dotr2.eulerGamma + log(1.0 / (abs(Dt_nonoise) + std::numeric_limits<double>::epsilon()));

                    //coeff0 = Kernel*Signal_profile;
                    Mat<T> coeff0 = Kernel * Signal_profile;

                    // coeff0 = coeff0./repmat(coeff0(1,:),[size(coeff0,1) 1]);
                    coeff0 = coeff0 / repmat(coeff0.row(0), size(coeff0, 0), 1);

                    //K_dot_r2(1) = (4/pi)*(2*sqrt(pi))/16;
                    K_dot_r2(0) = (4.0 / datum::pi) * (2.0 * std::sqrt(datum::pi)) / 16.0;

                    //ss = coeff0.*repmat(K_dot_r2,[1 size(coeff0,2)]);
                    ss = coeff0 % repmat(K_dot_r2, 1, size(coeff0, 1));
                    // ODF = basisV*ss;
                    ODF = basisV * ss;

                    // ODF = ODF - repmat(min(ODF),size(V,1),1);
                    ODF = ODF - repmat(min(ODF), size(V, 0), 1);

                    if (opts.norm_odfs)    {
                        // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                        ODF = ODF / repmat(sum(ODF), size(V, 0), 1);
                    }

                    //                    Mat<T> TempMat = trans(basisV)*basisV;
                    //                    ss = inv(TempMat)*trans(basisV)*ODF;
                }
                    break;
                case QBI_CSA: {
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

                    tempSignal = tempSignal / (tempS0 + std::numeric_limits<double>::epsilon());
                    tempSignal.elem(find(tempSignal > 1.0)).ones();
                    tempSignal.elem(find(tempSignal < 0.0)).zeros();

                    // Signal_profile = log(-log(tempSignal./tempS0));
                    Mat<T> Signal_profile = log(-log(tempSignal)+ std::numeric_limits<double>::epsilon());
                    // coeff0 = Kernel*Signal_profile;
                    Mat<T> coeff0 = Kernel * Signal_profile;

                    //coeff0(1,:) = ones(1,size(coeff0,2));
                    coeff0.row(0) = ones<Row<T>>(size(coeff0, 1));

                    // ss = coeff0.*repmat(K_csa,[1 size(coeff0,2)]);
                    ss = coeff0 % repmat(K_csa, 1, size(coeff0, 1));

                    // ODF = basisV*ss;
                    ODF = basisV * ss;
                    //ODF = ODF - repmat(min(ODF),size(V,1),1);
                    ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);
                    if (opts.norm_odfs)    {
                        // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                        ODF = ODF / repmat(sum(ODF, 0), size(V, 0), 1);
                    }

                }
                    break;
                case GQI_L1: {

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
                    //end
                    //tempSignal = tempSignal./tempS0;
                    tempSignal = tempSignal / (tempS0 + std::numeric_limits<double>::epsilon());
                    tempSignal.elem(find(tempSignal > 1.0)).ones();
                    tempSignal.elem(find(tempSignal < 0.0)).zeros();

                    //ODF = Kernel*tempSignal;
                    ODF = Kernel * tempSignal;

                    //ODF = ODF - repmat(min(ODF),size(V,1),1);
                    ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);

                    if (opts.norm_odfs)    {
                        //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                        ODF = ODF / repmat(sum(ODF, 0), size(V, 0), 1);
                    }

                    //                    % Computing SH Coefficients
                    //                    ss = zeros(Nsh,Nv);
                    //                    ss = inv(basisV'*basisV)*basisV'*ODF;

                    //                     Mat<T> TempMat = trans(basisV)*basisV;
                    //                     ss = inv(TempMat)*trans(basisV)*ODF;

                }
                    break;
                case GQI_L2: {
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
                    //end
                    //tempSignal = tempSignal./tempS0;
                    tempSignal = tempSignal / (tempS0 + std::numeric_limits<double>::epsilon());
                    tempSignal.elem(find(tempSignal > 1.0)).ones();
                    tempSignal.elem(find(tempSignal < 0.0)).zeros();

                    //ODF = Kernel*tempSignal*opts.gqi_l2.lambda^3/pi;
                    ODF = Kernel * tempSignal * pow(opts.gqi.lambda, 3) / datum::pi;

                    //ODF = ODF - repmat(min(ODF),size(V,1),1);
                    ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);
                    if (opts.norm_odfs)    {
                        //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                        ODF = ODF / repmat(sum(ODF, 0), size(V, 0), 1);
                    }
                    //                    Mat<T> TempMat = trans(basisV)*basisV;
                    //                    ss = inv(TempMat)*trans(basisV)*ODF;

                }
                    break;
                case QBI: {
                    // coeff = Kernel*diffSignal(indb1,:);
                    Mat<T> coeff0  = Kernel * diffSignal.rows(indb1);

                    // ss = coeff.*repmat(K,[1 size(coeff,2)]);
                    ss = coeff0 % repmat(K, 1, size(coeff0, 1));

                    // ODF = basisV*ss;
                    ODF = basisV * ss;

                    // ODF = ODF - repmat(min(ODF),size(V,1),1);
                    ODF = ODF - repmat(min(ODF), size(V, 0), 1);
                    if (opts.norm_odfs)    {
                        // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                        ODF = ODF / repmat(sum(ODF), size(V, 0), 1);
                    }
                }
                    break;
                case RUMBA_SD:{
                    // fODF0 = ones(size(Kernel,2),1);
                    Mat<T> fODF0(Kernel.n_cols, 1);
                    T mean_SNR;

                    // fODF0 = fODF0/sum(fODF0); % Normalizing ODFs values
                    fODF0.fill(1.0 / fODF0.n_elem);

                    LOG_INFO << "calling intravox_fiber_reconst_sphdeconv_rumba_sd";
                    // ODF = Intravox_Fiber_Reconst_sphdeconv_rumba_sd(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter); % Intravoxel Fiber Reconstruction using RUMBA (Canales-Rodriguez, et al 2015)
                    ODF = intravox_fiber_reconst_sphdeconv_rumba_sd<T>(diffSignal, Kernel, fODF0,
                                                                       opts.rumba_sd.Niter, mean_SNR);

                    if (opts.norm_odfs)    {
                        // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                        ODF = ODF / repmat(sum(ODF,0) + std::numeric_limits<double>::epsilon(), ODF.n_rows  ,1);
                    }

                    LOG_INFO << "Estimated mean SNR = " << mean_SNR;

                    Row<T> tfSum = sum(ODF,0);

#pragma omp parallel for
                    for (uword i = 0; i < inda.n_elem; ++i) {
                        slicevf_CSF.at(inda(i)) = ODF.at(ODF.n_rows - 2, i)/tfSum.at(i);
                    }

#pragma omp parallel for
                    for (uword i = 0; i < inda.n_elem; ++i) {
                        slicevf_GM.at(inda(i)) = ODF.at(ODF.n_rows - 1, i)/tfSum.at(i);
                    }


                    //ODF_iso = ODF(end,:) + ODF(end-1,:);
                    ODF_iso = ODF.row(ODF.n_rows - 1) + ODF.row(ODF.n_rows - 2);

                    // ODF = ODF(1:end-2,:);
                    ODF.resize(ODF.n_rows - 2, ODF.n_cols);

                    //volvf_WM(inda) = sum(ODF,1);    % Volume fraction of WM
                    Row<T> tsum = sum(ODF, 0);
#pragma omp parallel for
                    for (uword i = 0; i < inda.n_elem; ++i) {
                        slicevf_WM.at(inda.at(i)) = tsum.at(i)/tfSum.at(i);
                    }

                    // Adding the isotropic components to the ODF
                    ODF_WM = ODF;
                    ODF = ODF + repmat(ODF_iso / ODF.n_rows, ODF.n_rows, 1);
                   // ODF = ODF / repmat(sum(ODF, 0) + std::numeric_limits<double>::epsilon(), ODF.n_rows, 1);

                    //                    Mat<T> TempMat = trans(basisV)*basisV;
                    //                    ss = inv(TempMat)*trans(basisV)*ODF;

                    break;
                }
                    break;
                }
                if (opts.reconsMethod != DTI_NNLS){
                    Row<T> temp = stddev(ODF, 0, 0) /
                            sqrt(mean(pow(ODF, 2), 0) + std::numeric_limits<double>::epsilon());

                    Mat<T> TempMat = trans(basisV)*basisV;
                    ss = inv(TempMat)*trans(basisV)*(ODF);
                    ss_wm = inv(TempMat)*trans(basisV)*ODF_WM;
                    if (opts.norm_odfs)    {
                        ss =ss/repmat(sqrt(sum(pow(ss, 2)))+ std::numeric_limits<double>::epsilon(),Nsh,1);
                        ss_wm =ss_wm/repmat(sqrt(sum(pow(ss_wm, 2)))+ std::numeric_limits<double>::epsilon(),Nsh,1);
                    }
                    // %% =================== Estimating ODFs Peaks ===========================
                    // #pragma omp parallel for
                    Mat<T> peaksMat(Nd,inda.n_elem);
                    for (int i = 0; i < Nd; ++i) {
                        uvec tempIndexODF = neighMat(i,0);
                        //                        LOG_INFO << neighMat(i,0);

                        Mat<T> peakDiff = repmat(ODF.row(i),tempIndexODF.n_elem,1) - ODF.rows(tempIndexODF);
                        //                        LOG_INFO <<peakDiff;

                        umat A = all(peakDiff > 0);
                        peaksMat.row(i) = ODF.row(i)%A;
                        //                        LOG_INFO << i+1;
                        //                        LOG_INFO << A;
                    }
                    peaksMat = peaksMat/repmat(sum(peaksMat)+ std::numeric_limits<double>::epsilon(),Nd,1);
                    // %% ====================================================================
                    //                    V.col(1) = -1*V.col(1);
                    //                    V.col(2) = -1*V.col(2);
                    for (int i = 0; i < inda.n_elem; ++i) {
                        Col<T> tempCol = peaksMat.col(i);

                        uvec sortIndex = sort_index(tempCol,"descend");
                        //                        LOG_INFO << sortIndex;

                        uvec tt = sortIndex.rows(0,Np-1);
                        //                        LOG_INFO << V.rows(tt);

                        Mat<T> mat2Write = trans(V.rows(tt) % repmat(tempCol.rows(tt),1,3));


                        //                        LOG_INFO << repmat(tempCol.rows(tt),1,3);

                        //                        LOG_INFO << vectorise(mat2Write);
                        ordpeaksMat.insert_cols(i,vectorise(mat2Write));
                        //                        LOG_INFO << ordpeaksMat.col(i) ;

                    }


                    //                        LOG_INFO <<ordpeaksMat;



#pragma omp parallel for
                    for (uword i = 0; i < inda.n_elem; ++i) {
                        slice_GFA.at(inda.at(i)) = temp.at(i);
                    }
                    //                    LOG_INFO << slice_GFA;
                }

                // % Allocating memory to save the ODF
                // globODFslice = zeros(xdiff,ydiff,Nd);

                // % Reordering ODF
                //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                Mat<uword> allIndexesODF = repmat(inda.t(), ODF.n_rows, 1);

                // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                Mat<uword> ODFindexes = allIndexesODF + totalNvoxels *
                        repmat(linspace<uvec>(0, ODF.n_rows - 1, ODF.n_rows), 1,
                               inda.n_elem);




                if (opts.reconsMethod != DTI_NNLS) {


                    // % Reordering SH
                    //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                    Mat<uword> allIndexesSH = repmat(inda.t(), Nsh, 1);

                    // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                    Mat<uword> SHindexes = allIndexesSH + totalNvoxels *
                            repmat(linspace<uvec>(0, Nsh - 1, Nsh), 1,
                                   inda.n_elem);

                    // % Reordering Peaks
                    //  allIndexesODF = repmat(inda(:)',[size(ODF,1) 1]); % Image indexes
                    Mat<uword> allIndexesPeaks = repmat(inda.t(), Np*3, 1);

                    // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
                    Mat<uword> Peaksindexes = allIndexesPeaks + totalNvoxels *
                            repmat(linspace<uvec>(0, Np*3 - 1, Np*3), 1,
                                   inda.n_elem);



                    // globODFslice(ODFindexes(:)) = ODF(:);

                    if (opts.save_mrtrix) {
#pragma omp parallel for
                        for (uword j = 0; j < ss.n_cols; ++j) {
                            for (uword i = 0; i < ss.n_rows; ++i) {
                                globSHWMslice.at(SHindexes.at(i, j)) = ss_wm.at(i, j);
                                globSHslice.at(SHindexes.at(i, j)) = ss.at(i, j);
                            }
                        }
                    } else {
#pragma omp parallel for
                        for (uword j = 0; j < ODF.n_cols; ++j) {
                            for (uword i = 0; i < ODF.n_rows; ++i) {
                                globODFslice.at(ODFindexes.at(i, j)) = ODF.at(i, j);
                            }
                        }
                    }

                    if (opts.save_peaks) {
                        for (uword j = 0; j < ordpeaksMat.n_cols; ++j) {
                            for (uword i = 0; i < ordpeaksMat.n_rows; ++i) {
                                globPeaksslice.at(Peaksindexes.at(i, j)) = ordpeaksMat.at(i, j);
                            }
                        }
                    }

                    //                    #pragma omp parallel for
                    for (uword i = 0; i < xdiff; ++i) {
                        Index4DType coord;
                        coord[0] = i;
                        coord[2] = slice;
                        for (uword j = 0; j < ydiff; ++j) {
                            coord[1] = j;

                            if (opts.save_mrtrix) {
                                for (uword k = 0; k < Nsh; ++k) {
                                    coord[3] = k;
                                    imageSH->SetPixel(coord, globSHslice.at(i, j, k));
                                    imageSH_WM->SetPixel(coord, globSHWMslice.at(i, j, k));
                                }
                            } else {
                                for (uword k = 0; k < Nd; ++k) {
                                    coord[3] = k;
                                    imageODF->SetPixel(coord, globODFslice.at(i, j, k));
                                }
                            }

                            if (opts.save_peaks) {
                                for (uword k = 0; k < Np*3; ++k) {
                                    coord[3] = k;
                                    imagePeaks->SetPixel(coord, globPeaksslice.at(i, j, k));
                                }
                            }
                        }
                    }

#pragma omp parallel for
                    for (uword j = 0; j < ydiff; ++j) {
                        Index3DType coord;
                        coord[1] = j;
                        coord[2] = slice;
                        for (uword i = 0; i < xdiff; ++i) {
                            coord[0] = i;
                            if (opts.reconsMethod == RUMBA_SD) {

                                imageCSF->SetPixel(coord, slicevf_CSF.at(i, j));
                                imageGM->SetPixel(coord, slicevf_GM.at(i, j));
                                imageWM->SetPixel(coord, slicevf_WM.at(i, j));
                            }
                            imageGFA->SetPixel(coord, slice_GFA.at(i, j));
                        }
                    }
                }
            }
        }
    }
        break;
    case VOLUME: {

        Cube<T> slicevf_CSF(xdiff, ydiff, zdiff, fill::zeros);
        Cube<T> slicevf_GM(xdiff, ydiff, zdiff, fill::zeros);
        Cube<T> slicevf_WM(xdiff, ydiff, zdiff, fill::zeros);
        Cube<T> slicevf_GFA(xdiff, ydiff, zdiff, fill::zeros);
        Row<T> ODF_iso(ydiff, fill::zeros);

        std::vector<Cube<T>> globODFslice;
        for (uword i = 0; i < Nd; ++i) {
            Cube<T> tmp(xdiff, ydiff, zdiff, fill::zeros);
            globODFslice.push_back(tmp);
        }

        // Imask = logical(Imask);
        Vmask.elem(find(Vmask != 0.0)).ones();

        if (accu(Vmask) != 0) {
            Cube<T> S0_est_M(xdiff, ydiff, zdiff, fill::zeros);
            // isolating the b0 images
            // S0_est = squeeze( Idiff(:,:,indb0) );
            std::vector<Cube<T>> S0_est;

            for (uword i = 0; i < indb0.n_elem; ++i)
                S0_est.push_back(Vdiff[i]);

            // if length(indb0) > 1
            //    S0_est = mean(S0_est,4).*Imask;
            // end

            if (indb0.n_elem == 1) {
                S0_est_M = S0_est[0];
            } else {
                for (uword i = 0; i < indb0.n_elem; ++i)
                    S0_est_M = S0_est_M + S0_est[i];
                S0_est_M = S0_est_M / indb0.n_elem;
                S0_est_M = S0_est_M % Vmask;
            }

            // reordering the data such that the b0 image appears first (see lines 152-166)
            // Idiff(:,:,:,indb0) = [];
            for (int i = 0; i < indb0.n_elem; ++i)
                Vdiff[indb0(i)].reset();

            for (sword graddir = Ngrad - 1; graddir >= 0; --graddir)
                if (Vdiff[graddir].n_elem == 0) Vdiff.erase(Vdiff.begin() + graddir);

            int pSize = Vdiff.size();

            //Idiff = cat(4,S0_est,Idiff);
            Vdiff.resize(pSize + indb0.n_elem);
            for (sword i = pSize - 1; i >= 0; --i)
                Vdiff[i + indb0.n_elem] = Vdiff[i];

            for (uword i = 0; i < indb0.n_elem; ++i)
                Vdiff[i] = S0_est[i];

            // normalize the signal S/S0
            // for graddir = 1:Ngrad_mod
#pragma omp parallel for
            for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                //Idiff(:,:,:,graddir) = squeeze(Idiff(:,:,:,graddir).* Imask)./(S0_est + eps);
                Vdiff[graddir] = Vdiff[graddir] / (S0_est_M + std::numeric_limits<double>::epsilon());
            }

            // Repair the signal
            // Idiff(Idiff>1) = 1;
            // Idiff(Idiff<0) = 0;
#pragma omp parallel for
            for (uword graddir = 0; graddir < Ngrad; ++graddir) {
                Vdiff[graddir].elem(find(Vdiff[graddir] > 1.0)).ones();
                Vdiff[graddir].elem(find(Vdiff[graddir] < 0.0)).zeros();
            }
            //inda = find(squeeze(Idiff(:,:,1))>0);
            uvec inda_vec = find(Vmask != 0);
            Mat<uword> inda = conv_to<Mat<uword>>::from(inda_vec);

            //totalNvoxels = prod(size(Imask));
            size_t totalNvoxels = Vmask.n_elem;

            //allIndexes = repmat(inda(:)',[Ngrad 1]);
            Mat<uword> allIndixes = repmat(inda.t(), Ngrad, 1);
            //allIndixes.save("inda.txt",arma::raw_ascii);

            // diffSignal = Idiff(allIndexes + totalNvoxels*repmat([0:Ngrad-1]',[1 length(inda) ])); % Indexes in 4D
            Mat<uword> ind = allIndixes +
                    (totalNvoxels * repmat(linspace<Mat<uword>>(0, Ngrad - 1, Ngrad), 1, inda.n_elem));
            Mat<T> diffSignal(Ngrad, inda.n_elem);

#pragma omp parallel for
            for (uword j = 0; j < inda.n_elem; ++j)
                for (uword i = 0; i < Ngrad; ++i)
                    diffSignal.at(i, j) = Vdiff[i].at(inda_vec.at(j));  //Vdiff[i].at(ind(i,j));

            Mat<T> ODF;
            switch (opts.reconsMethod) {
            case DTI_NNLS:
                break;
            case QBI_DOTR2: {
                uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
                uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);

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
                Mat<T> tempVar =
                        -log(tempSignal / tempS0) / repmat(diffBvals.rows(indb1), 1, size(tempSignal, 1));

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
                K_dot_r2(0) = (4.0 / datum::pi) * (2.0 * std::sqrt(datum::pi)) / 16.0;

                //ss = coeff0.*repmat(K_dot_r2,[1 size(coeff0,2)]);
                Mat<T> ss = coeff0 % repmat(K_dot_r2, 1, size(coeff0, 1));

                // ODF = basisV*ss;
                ODF = basisV * ss;

                // ODF = ODF - repmat(min(ODF),size(V,1),1);
                ODF = ODF - repmat(min(ODF), size(V, 0), 1);

                if (opts.norm_odfs)    {
                    // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                    ODF = ODF / repmat(sum(ODF), size(V, 0), 1);
                }
            }
                break;
            case QBI_CSA: {
                uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
                uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);

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

                // Signal_profile = log(-log(tempSignal./tempS0));
                Mat<T> Signal_profile = log(-log(tempSignal / tempS0));
                // coeff0 = Kernel*Signal_profile;
                Mat<T> coeff0 = Kernel * Signal_profile;

                //coeff0(1,:) = ones(1,size(coeff0,2));
                coeff0.row(0) = ones<Row<T>>(size(coeff0, 1));

                // ss = coeff0.*repmat(K_csa,[1 size(coeff0,2)]);
                Mat<T> ss = coeff0 % repmat(K_csa, 1, size(coeff0, 1));

                // ODF = basisV*ss;
                ODF = basisV * ss;

                // ODF = ODF - repmat(min(ODF),size(V,1),1);
                ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);
                if (opts.norm_odfs)    {
                    // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                    ODF = ODF / repmat(sum(ODF, 0), size(V, 0), 1);
                }
            }
                break;
            case DSI:
                break;
            case GQI_L1: {
                uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
                uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);

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
                //end
                //tempSignal = tempSignal./tempS0;
                tempSignal = tempSignal / tempS0;

                //ODF = Kernel*tempSignal;
                ODF = Kernel * tempSignal;

                //ODF = ODF - repmat(min(ODF),size(V,1),1);
                ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);
                if (opts.norm_odfs)    {
                    //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                    ODF = ODF / repmat(sum(ODF, 0), size(V, 0), 1);
                }
            }
                break;
            case GQI_L2: {
                uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
                uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);

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
                //end
                //tempSignal = tempSignal./tempS0;
                tempSignal = tempSignal / tempS0;

                //ODF = Kernel*tempSignal*opts.gqi_l2.lambda^3/pi;
                ODF = Kernel * tempSignal * pow(opts.gqi.lambda, 3) / datum::pi;

                //ODF = ODF - repmat(min(ODF),size(V,1),1);
                ODF = ODF - repmat(min(ODF, 0), size(V, 0), 1);
                if (opts.norm_odfs)    {
                    //ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                    ODF = ODF / repmat(sum(ODF, 0), size(V, 0), 1);
                }
            }
                break;
            case QBI: {
                Mat<T> coeff, ss;

                uvec indb0 = find((prod((sum(abs(diffGrads),1), diffBvals),1) ==0) || diffBvals <=10);
                uvec indb1 = find((prod((sum(abs(diffGrads),1), diffBvals),1) !=0) && diffBvals > 10);

                // coeff = Kernel*diffSignal(indb1,:);
                coeff = Kernel * diffSignal.rows(indb1);

                // ss = coeff.*repmat(K,[1 size(coeff,2)]);
                ss = coeff % repmat(K, 1, size(coeff, 1));

                // ODF = basisV*ss;
                ODF = basisV * ss;
                // ODF = ODF - repmat(min(ODF),size(V,1),1);
                ODF = ODF - repmat(min(ODF), size(V, 0), 1);
                if (opts.norm_odfs)    {
                    // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                    ODF = ODF / repmat(sum(ODF), size(V, 0), 1);
                }
            }
                break;
            case RUMBA_SD:
                // fODF0 = ones(size(Kernel,2),1);
                Mat<T> fODF0(Kernel.n_cols, 1);
                T mean_SNR;

                // fODF0 = fODF0/sum(fODF0); % Normalizing ODFs values
                fODF0.fill(1.0 / fODF0.n_elem);

                LOG_INFO << "calling intravox_fiber_reconst_sphdeconv_rumba_sd";
                // ODF = Intravox_Fiber_Reconst_sphdeconv_rumba_sd(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter); % Intravoxel Fiber Reconstruction using RUMBA (Canales-Rodriguez, et al 2015)
                ODF = intravox_fiber_reconst_sphdeconv_rumba_sd<T>(diffSignal, Kernel, fODF0,
                                                                   opts.rumba_sd.Niter, mean_SNR);
                if (opts.norm_odfs)    {
                    // ODF = ODF./repmat(sum(ODF),size(V,1),1); % normalization
                    ODF = ODF / repmat(sum(ODF,0) + std::numeric_limits<double>::epsilon(), ODF.n_rows  ,1);
                }
                // TODO ODF = intravox_fiber_reconst_sphdeconv_rumba_sd_gpu<T>(diffSignal, Kernel, fODF0, opts.rumba_sd.Niter);
                LOG_INFO << "Estimated mean SNR = " << mean_SNR;

                Row<T> tfSum = sum(ODF, 0);

#pragma omp parallel for
                for (uword i = 0; i < inda.n_elem; ++i) {
                    slicevf_CSF.at(inda.at(i)) = ODF.at(ODF.n_rows - 2, i)/tfSum.at(i);
                }

#pragma omp parallel for
                for (uword i = 0; i < inda.n_elem; ++i) {
                    slicevf_GM.at(inda.at(i)) = ODF.at(ODF.n_rows - 1, i)/tfSum.at(i);
                }

                //ODF_iso = ODF(end,:) + ODF(end-1,:);
                ODF_iso = ODF.row(ODF.n_rows - 1) + ODF.row(ODF.n_rows - 2);

                // ODF = ODF(1:end-2,:);
                ODF.resize(ODF.n_rows - 2, ODF.n_cols);

                //volvf_WM(inda) = sum(ODF,1);    % Volume fraction of WM
                Row<T> tsum = sum(ODF, 0);
#pragma omp parallel for
                for (uword i = 0; i < inda.n_elem; ++i) {
                    slicevf_WM.at(inda.at(i)) = tsum.at(i)/tfSum.at(i);
                }

                // Adding the isotropic components to the ODF
                ODF = ODF + repmat(ODF_iso / ODF.n_rows, ODF.n_rows, 1);
                //ODF = ODF / repmat(sum(ODF, 0) + std::numeric_limits<double>::epsilon(), ODF.n_rows, 1);

                // std(ODF,0,1)./( sqrt(mean(ODF.^2,1)) + eps )
                Row<T> temp = stddev(ODF, 0, 0) /
                        sqrt(mean(pow(ODF, 2), 0) + std::numeric_limits<double>::epsilon());
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
            Mat<uword> allIndexesODF = repmat(inda.t(), ODF.n_rows, 1);

            // ODFindexes = allIndexesODF + totalNvoxels*repmat([0:size(ODF,1)-1]',[1 length(inda) ]); % Indexes in 4D
            Mat<uword> ODFindexes = allIndexesODF + totalNvoxels *
                    repmat(linspace<Mat<uword>>(0, ODF.n_rows - 1, ODF.n_rows),
                           1, inda.n_elem);

            if (opts.reconsMethod != DTI_NNLS) {
                // globODFslice(ODFindexes(:)) = ODF(:);
                uword slice_size = xdiff * ydiff * zdiff;
#pragma omp parallel for
                for (uword j = 0; j < ODF.n_cols; ++j) {
                    for (uword i = 0; i < ODF.n_rows; ++i) {
                        globODFslice[i].at(ODFindexes(i, j) % slice_size) = ODF.at(i, j);
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
                        T temp = tempc.at(i, j, k);
                        coord[2] = k;
                        imageODF->SetPixel(coord, temp);
                    }
                }
            }
        }
    }
        break;
    }

    if (opts.reconsMethod != DTI_NNLS) {

        if (opts.save_mrtrix) {
            LOG_INFO << "writting SH file " << filenameSH;
            WriteImage<Image4DType, NiftiType>(filenameSH, imageSH);
            WriteImage<Image4DType, NiftiType>(filenameSH_WM, imageSH_WM);
        } else {
            LOG_INFO << "writting ODF file " << ODFfilename;
            WriteImage<Image4DType, NiftiType>(ODFfilename, imageODF);
        }

        if (opts.save_peaks) {
            LOG_INFO << "writting Peaks file " << filenamePeaks;
            WriteImage<Image4DType, NiftiType>(filenamePeaks, imagePeaks);
        }

        LOG_INFO << "writting GFA file " << filenameGFA;
        // LOG_INFO <<  imageGFA;
        WriteImage<Image3DType, NiftiType>(filenameGFA, imageGFA);

        if (opts.reconsMethod == RUMBA_SD) {
            LOG_INFO << "writting CSF file " << filenameCSF;
            // LOG_INFO << imageCSF;
            WriteImage<Image3DType, NiftiType>(filenameCSF, imageCSF);

            LOG_INFO << "writting GM file " << filenameGM;
            // LOG_INFO <<  imageGM;
            WriteImage<Image3DType, NiftiType>(filenameGM, imageGM);

            LOG_INFO << "writting WM file " << filenameWM;
            // LOG_INFO <<  imageWM;
            WriteImage<Image3DType, NiftiType>(filenameWM, imageWM);
        }
    }
}
}

#endif

