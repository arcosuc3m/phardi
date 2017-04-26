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

#ifndef COMMON_H
#define COMMON_H

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1

#include <armadillo>
#include <math.h>
#include <cmath>
#include <limits>
#include <arrayfire.h>

#define fftshift3(in)  af::shift(in, in.dims(0)/2, in.dims(1)/2, in.dims(2)/2)
#define ifftshift3(in) af::shift(in, (in.dims(0)+1)/2, (in.dims(1)+1)/2, (in.dims(2)+1)/2)
#define fft3(in) af::fft3(in)

namespace phardi {


    template <typename T>
    void Cart2Sph(T x, T y, T z, T & phi, T & theta)
    {
        T hypotxy;
        hypotxy = std::sqrt(std::pow(x,2) + std::pow(y,2));

        // compute elev
        theta = std::atan2(z, hypotxy);

        // compute az
        phi = std::atan2(y, x);
    }

    template <typename T>
    inline T sinc(const T x)
    {
        if (x==0)
            return 1;
        return std::sin(x)/x;
    }


    template <typename T>
    inline T factorial(T n)
    {
        if (n == 0)
            return 1;
        return n * factorial(n - 1);
    }

    template <typename T>
    arma::Mat<T> legendre (arma::uword n, arma::Col<T>& x) {
        using namespace arma;

        if(n == 0) {
            Mat<T> res;
            res.resize(x.n_elem, 1);
            res.ones();
            return res;
        } else {
            Col<T> rootn(2*n+1);
            for (uword i = 0; i <= 2*n; ++i)
                rootn(i) = std::sqrt(i);

            Col<T> s = sqrt(1 - pow(x,2));
            Mat<T> P(n+3, x.n_elem);
	    P.zeros();          

            Col<T> twocot = -2.0 * x / s;

            Col<T> sn = pow ( -1.0 * s, n);
            T tol = std::sqrt(std::numeric_limits<T>::min());
            uvec ind = find( s>0 && abs(sn)<=tol);
            uvec nind = find( x!=1 && abs(sn)>=tol);
  
            if (nind.n_elem > 0) {
                Col<T> d(n);
                for (uword i = 0; i < n; ++i)
                    d(i) = 2*(i+1);

                T c = prod(1-(1/d));
    
                P.row(n) = std::sqrt(c) * sn(nind).t();
                P.row(n-1) = P.row(n) % twocot(nind).t() * n / rootn(rootn.n_elem - 1);
                for (sword m = n-2; m >=0 ; --m) {
                    P.row(m) = (P.row(m+1) % twocot(nind).t()*(m+1) - P.row(m+2)*rootn(n+m+2)*rootn(n-m-1))/(rootn(n+m+1)*rootn(n-m)); 
                }

            }
            
            Mat<T> y = P.submat(0,0,n+1,P.n_cols-1);
            uvec s0 = find(s == 0);
            if (s0.n_elem !=0)
                y.row(0) = pow(x(s0),n);
            for (uword m = 1; m <= n-1; ++m)
                y.row(m) = prod(rootn.subvec(n-m+1,n+m))*y.row(m);     
            y.row(n) = prod(rootn.subvec(1,rootn.n_elem-1)) * y.row(n);
            y.resize(n+1, x.n_elem);          
            y = y.t();
            return y;          
        }
    }


    constexpr arma::uword factorial(arma::uword n)
    {
        return n <= 1? 1 : (n * factorial(n - 1));
    }

    // Construct spherical harmonics basis matrix at specified points.
    // SYNTAX: [basis] = construct_SH_basis(degree, sphere_points, dl, real_or_complex);
    //
    // INPUTS:
    //
    // degree                - maximum degree of spherical harmonics;
    // sphere_points         - n sample points on sphere;          [nx3]
    // dl                    - {1} for full band; 2 for even order only
    // real_or_complex       - [{'real'} or 'complex'] basis functions.
    //
    // OUTPUTS:
    //
    // basis   - Spherical harmonics values at n points, [ n x k ]
    //           where k = (degree+1)^2 for full band or k = nchoosek(degree+2,2) for even only
    //
    // Example:
    //    sphere_points = rand(100,3);
    //    [basis] = construct_SH_basis(4, sphere_points, 2, 'real');

    template <typename T>
    void construct_SH_basis(arma::uword degree, const arma::Mat<T> & sphere_points, arma::uword dl, std::string real_or_complex, arma::Col<T>& theta, arma::Col<T>& phi, arma::Mat<T> & basis)
    {

        using namespace arma;

        T k, lconstant, precoeff;
        uword n;

        // n = size(sphere_points,1);  % number of points
        n = size(sphere_points, 0);

        phi.resize(n);
        theta.resize(n);

#pragma omp parallel for
        for (uword i = 0; i < n; ++i) {
            // [phi, theta] = cart2sph(sphere_points(:,1),sphere_points(:,2),sphere_points(:,3));
            Cart2Sph<T>(sphere_points(i,0), sphere_points(i,1), sphere_points(i,2), phi(i), theta(i)) ;
        }


        // theta = pi/2 - theta;  % convert to interval [0, PI]
        theta.transform([](T val) { return datum::pi/2.0 - val;});

        if (dl==1)
            k = (degree+1)*(degree+1) ;
        else
        {
            if (dl==2)
                k = (degree+2)*(degree+1)/2.0 ;
            else
                std::cerr << "dl can only be 1 or 2" << std::endl ;
        }

        Mat<T> Y(n, k, fill::zeros);

        for (uword l=0; l <= degree; l+=dl)
        {
            uword center;

            Col<T> ctheta = cos(theta);
            // Pm = legendre(l,cos(theta')); % legendre part
            Mat<T> Pm = legendre(l, ctheta);

            // lconstant = sqrt((2*l + 1)/(4*pi));
            lconstant = std::sqrt((2.0*l+1.0) / (4.0 * datum::pi));

            if (dl==2)
            {
                center = (l+1)*(l+2)/2-l ;
            }
            else
            {
                if (dl==1)
                {
                    center = (l+1)*(l+1)-l ;
                }
            }

            // Y(:,center) = lconstant*Pm(:,1);
            Y.col(center-1) = lconstant * Pm.col(0);

            for (uword m=1; m<=l; ++m)
            {
                // precoeff = lconstant * sqrt(factorial(l - m)/factorial(l + m));
                precoeff = lconstant * std::sqrt(factorial<T>(l-m)/factorial<T>(l+m)) ;

                if ("real" == real_or_complex)
                {
                    if (m % 2 == 1)
                    {
                        precoeff = -precoeff;
                    }
                    // Y(:, center + m) = sqrt(2)*precoeff*Pm(:,m+1).*cos(m*phi);
                    Y.col(center + m -1) = sqrt(2.0f) * precoeff * Pm.col(m) % cos(m*phi);

                    // Y(:, center - m) = sqrt(2)*precoeff*Pm(:,m+1).*sin(m*phi);
                    Y.col(center - m -1) = sqrt(2.0f) * precoeff * Pm.col(m) % sin(m*phi);
                }
                else if ("complex" == real_or_complex)
                {
                    if (m % 2 == 1)
                    {
                        //Y.cols(center+m) = precoeff * Pm.cols(m+1) %arma::exp(i*m*phi);
                        //Y.cols(center-m) = -(precoeff)*Pm.cols(m+1)%arma::exp(-(i)*m*phi) ;
                    }
                    else
                    {
                        //Y.cols(center+m) = precoeff*Pm.cols(m+1)%arma::exp(i*m*phi) ;
                        //Y.cols(center-m) = precoeff*Pm.cols(m+1)%arma::exp(-(i)*m*phi) ;
                    }
                }
                else
                {
                    std::cerr << "The last argument must be either \"real\" (default) or \"complex\"." << std::endl ;
                }
            }
        }
        basis = Y;
    }

    // This function computes the maximum order for the spherical harmonic decomposition (Lmax)
    // by taking into account the dimensionality of the signal (number of discrete points)
    // The above relationship is based on the real spherical harmonic expansion
    // Nmin is the minimum number of points necessary to use Lmax

    // Erick Jorge Canales-Rodríguez & Lester Melie-García

    template <typename T>
    void obtain_Lmax(const arma::Mat<T> & DirSignal, arma::uword & Lmax, arma::uword & Nmin)
    {
        using namespace arma;

        uword ind = 0;
        Mat<T> Signal(0,3);

        // pos = ismember(DirSignal, [0 0 0], 'rows');
        // ind = find(pos);
        // DirSignal(ind,:) = [];

        for (uword i = 0; i < DirSignal.n_rows; ++i) {
            if (!(DirSignal(i,0) == 0 && DirSignal(i,1) == 0 && DirSignal(i,2) == 0)) {
                Signal.insert_rows(ind,DirSignal.row(i));
                ind++;
            }
        }

        // N = length(DirSignal);
        uword N = Signal.n_rows;

        // Lmax = 0;
        Lmax = 0;

        // while (Lmax+1)*(Lmax+2)/2 <= N
        while ((Lmax+1) * (Lmax+2)/2.0 <= N)
        {
            // Lmax = Lmax + 2;
            Lmax += 2;
        }
        //Lmax = Lmax - 2;
        Lmax = Lmax-2 ;

        // Nmin = sum(1:4:2*Lmax+1);
        Nmin = arma::sum(regspace<Row<T>>(0, 4, 2*Lmax)) ;
    }

    // Project:   High Angular Resolution Diffusion Imaging Tools
    // Function to compute the hardi-matrix used in the spherical harmonic inversion
    // ---------------------
    // A = inv(Y'*Y + Lambda*Laplacian)*Y';
    // where Y = Ymatrix
    // ---------------------
    // Language:  MATLAB(R)
    //  Author:  Erick Canales-Rodríguez, Lester Melie-García, Yasser Iturria-Medina, Yasser Alemán-Gómez
    //  Date: 2013, Version: 1.2
    //
    // See also test_DSI_example, test_DOT_example, test_QBI_example,
    // test_DOT_R1_example, test_DOT_R2_vs_CSA_QBI_example.

    template <typename T>
    arma::Mat<T> recon_matrix(const arma::Mat<T> & Y, const arma::Mat<arma::uword> & L, T Lambda)
    {
        using namespace arma;

        Mat<T> YY;
        Mat<T> A(Y.n_cols,Y.n_rows);
        uword N;

        // YY = Y'*Y;
        YY = Y.t() * Y;
        // A = (YY + Lambda*L)\Y';

        Mat<T> aa = YY + Lambda * conv_to<Mat<T>>::from(L);
        Mat<T> bb = Y.t();
        A = solve (aa,bb);
        return A;
    }

    // Point Spread Function (PSF), defined by the main lobe of the experimental PSF (truncated theoretical solution).
    //
    //  Language:  MATLAB(R)
    //   Author:  Erick Canales-Rodríguez, Lester Melie-García, Yasser Iturria-Medina, Yasser Alemán-Gómez
    //   Date: 2013, Version: 1.2
    //
    // --- Some comments ----
    // Notice that the theoretical/experimental PSF is just the fourier transform of the q-space points
    // where the signal was measured (cartesian sampling scheme). That is, the FFT of a 3D binary image
    // with values equal to 1 for those pixels where the signal was measured in q-space, and zero elsewhere.
    // However, the deconvolution using the resulting PSF is very unstable and noisy due to the many sidelobes
    // of this PSF (including negative sidelobes) and the low resolution of the data.
    // In addition, due to the discretization, the obtained PSF is not smooth.
    //
    // In order to improve the deconvolution procedure in DSI, a practical consideration that I have used
    // is to preserve only the main lobe of the PSF, which is positive, due to this part of the function is the main
    // responsible of the unwanted convolution effect in DSI.
    // Note that the truncation level (radius) depends on the experimental sampling scheme.
    // The truncation implemented here in this code is for a standard DSI sampling scheme
    // with 515 points and bmax=7000-9000 s/mm^2
    // If you are using a different sampling scheme, then a different truncation radius must be used for optimal results.
    //
    // A different alternative is to use instead the gaussian function that best fit the theoretical PSF.
    // Under this approximation the PSF is smooth and definite positive by definition. It does not contain unwanted oscillations.
    // We have found that the deconvolution using this PSF also produces high quality results.
    // This alternative PSF is computed by the function: create_gaussian_PSF
    //
    // See also create_gaussian_PSF, test_DSI_example, SignalMatrixBuilding.

    template <typename T>
    void create_mainlobe_PSF(const arma::Mat<arma::uword> & qc, arma::uword Resolution, arma::Cube<T> & PSF,  arma::Cube<T> Sampling_grid)
    {
        using namespace arma;

        // Sampling_grid = zeros(Resolution,Resolution,Resolution);
        Sampling_grid = zeros<Cube<T>>(Resolution, Resolution, Resolution);

        PSF.set_size(size(Sampling_grid));

        //n = length(qc);
        uword n = qc.n_rows;

        for (uword i=0; i < n; ++i)
        {
            // Sampling_grid(qc(i,1),qc(i,2),qc(i,3)) = 1;
            Sampling_grid(qc(i, 0), qc(i, 1), qc(i, 2)) = 1;
        }

        af::array Sampling_grid_af = af::array(Sampling_grid.n_rows,Sampling_grid.n_cols, Sampling_grid.n_slices, Sampling_grid.memptr()); 

        // --- Experimental PSF
        // PSF  = (real(fftshift(fftn(ifftshift(Sampling_grid)))));
        af::array PSF_af = af::real(fftshift3(fft3((ifftshift3(Sampling_grid_af)))));

        PSF_af.host(PSF.memptr());
 
        // PSF(PSF<0)=0;
        PSF.elem( find(PSF < 0.0) ).zeros();

        // --- Thresholding
        // PSF = PSF.*Sampling_grid; % Truncated PSF
        PSF = PSF % Sampling_grid;

        // PSF = PSF/sum(PSF(:));
        PSF = PSF / accu(PSF);
    }

    // Project:   High Angular Resolution Diffusion Imaging Tools
    // Function to create the 3D Cartesian signal-matrix required in DSI
    //
    // Language:  MATLAB(R)
    // Author:  Erick Canales-Rodríguez, Lester Melie-García, Yasser Iturria-Medina, Yasser Alemán-Gómez
    // Date: 2013, Version: 1.2
    //
    // See also, test_DSI_example, create_mainlobe_PSF, create_gaussian_PSF
    template <typename T>
    arma::Cube<T> SignalMatrixBuilding_Volume(const arma::Mat<arma::uword> & qc, const arma::Col<T> & value, const arma::uword Resolution)
    {
        using namespace arma;

        Cube<T> sq;
        cube res(Resolution,Resolution,Resolution);

        // indexes = sub2ind([Resolution Resolution Resolution],qc(:,1),qc(:,2),qc(:,3));
        Col<uword> indexes(qc.n_rows);
        for (uword i = 0; i < qc.n_rows; ++i) {
            indexes(i) = sub2ind(size(res), qc(i,0), qc(i,1), qc(i,2));
        }

        sq = arma::zeros<Cube<T>>(Resolution, Resolution, Resolution) ;

        //sq(indexes) = value;
        for (uword i = 0; i < value.n_elem; ++i)
            sq.at(indexes(i)) = value.at(i);

        return sq;
    }


    template <typename T>
    arma::SpMat<T> SignalMatrixBuilding_Volume(const arma::Mat<arma::uword> & qc, const arma::Mat<T> & value, const arma::uword Resolution)
    {
        using namespace arma;

        cube res(Resolution,Resolution,Resolution);
        // indexes = sub2ind([Resolution Resolution Resolution],qc(:,1),qc(:,2),qc(:,3));
        Col<uword> indexes(qc.n_rows);
        for (uword i = 0; i < qc.n_rows; ++i) {
            indexes(i) = sub2ind(size(res), qc(i,0), qc(i,1), qc(i,2));
        }

	uword totalNvoxels = pow(Resolution,3);

        // allIndexes = (repmat(indexes(:)',[size(value,2) 1]) + totalNvoxels*repmat([0:size(value,2)-1]',[1 length(indexes) ]))'; % Indexes in 4D

        Mat<uword> temp(value.n_cols,1);
        temp.col(0) = linspace<uvec>(0, value.n_cols-1);
        Mat<uword> allIndexes = (repmat(indexes.t(), value.n_cols, 1) + totalNvoxels * repmat(temp, 1 , indexes.n_elem)).t();
 
        // sq = sparse(totalNvoxels*size(value,2),1);
        SpMat<T> sq(totalNvoxels*value.n_cols,1);

        // sq(allIndexes(:)) = value(:);
        uvec allIndexes_v = vectorise(allIndexes);
        Col<T> value_v = vectorise(value);

	for (uword i = 0; i < totalNvoxels*value.n_cols; ++i)
            sq(allIndexes_v(i)) = value_v.at(i);  

        return sq;
    }

}

#endif

