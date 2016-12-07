/*
Copyright (c) 2016 
Javier Garcia Blas (fjblas@inf.uc3m.es)
Jose Daniel Garcia Sanchez (josedaniel.garcia@uc3m.es)
Yasser Aleman (yaleman@hggm.es)
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

#ifndef COMMON_H
#define COMMON_H

#include <armadillo>
#include <math.h>
#include <boost/math/special_functions/legendre.hpp>

namespace phardi {


    template <typename T>
    void Cart2Sph(T x, T y, T z, T & phi, T & theta)
    {
        double hypotxy;
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
    inline T span(size_t a, size_t b) {
        T s((arma::uword) 0);

        if (a > b) {
            s.set_size(1);
            s(0) = 1;
        }
        else {
            arma::uword n = b - a;
            s.set_size(n + 1);
            for (size_t ii = 0; ii <= n; ++ii)
                s(ii) = ii + a;
        }
        return s;
    }

    template <typename T>
    inline T span(size_t a, size_t step, size_t b)
    {
        T s;
        arma::uword n = (b - a + step) / step;
        s.set_size(n);
        for (int ii = 0; ii < n; ii++)
        {
            s(ii) = step * ii + a;
        }
        return s;
    }

    template <typename T>
    inline T factorial(T n)
    {
        if (n == 0)
            return 1;
        return n * factorial(n - 1);
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

        T k, lconstant, center, precoeff;
        uword n, l, m;

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

        for (l=0; l < degree; l+=dl)
        {
            Col<T> Pm(theta.n_elem);

#pragma omp parallel for
            for (uword i = 0; i < Pm.n_elem; ++i) {
                // Pm = legendre(l,cos(theta')); % legendre part
                Pm(i) = boost::math::legendre_p(l, std::cos(theta(i)));
            }

            // lconstant = sqrt((2*l + 1)/(4*pi));
            lconstant = std::sqrt((2*l+1) / (4 * datum::pi));

            if (dl==2)
            {
                center = (l+1)*(l+2)/2.0-l ;
            }
            else
            {
                if (dl==1)
                {
                    center = (l+1)*(l+1)-l ;
                }
            }

            // Y(:,center) = lconstant*Pm(:,1);
            Y.col(center) = lconstant * Pm;

            for (m=0; m<l; ++m)
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
#pragma omp parallel for
                    for (uword i = 0; i < phi.n_elem; ++i) {
                        Y(i, center + m) = std::sqrt(2) * precoeff * Pm(m + 1) * std::cos(m * phi(i));
                    }

                    // Y(:, center - m) = sqrt(2)*precoeff*Pm(:,m+1).*sin(m*phi);
#pragma omp parallel for
                    for (uword i = 0; i < phi.n_elem; ++i) {
                        Y(i, center - m) = std::sqrt(2) * precoeff * Pm(m + 1) * std::sin(m * phi(i));
                    }
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
        Nmin = arma::sum(span<Row<T>>(0, 4, 2*Lmax)) ;
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
    void create_mainlobe_PSF(const arma::Mat<T> qc, size_t Resolution, T& PSF,  arma::Cube<T>& Sampling_grid)
    {
        using namespace arma;

        // Sampling_grid = zeros(Resolution,Resolution,Resolution);
        Sampling_grid.set_size(Resolution, Resolution, Resolution);
        Sampling_grid.zeros();

        //n = length(qc);
        uword n = qc.n_rows;

#pragma omp parallel for
        for (uword i=0; i < n; ++i)
        {
            // Sampling_grid(qc(i,1),qc(i,2),qc(i,3)) = 1;
            Sampling_grid(qc(i, 0), qc(i, 1), qc(i, 2)) = 1 ;
        }

        // --- Experimental PSF
        // PSF  = (real(fftshift(fftn(ifftshift(Sampling_grid)))));
        PSF = (arma::real(ifft(fft(ifft(Sampling_grid))))) ;

        // PSF(PSF<0)=0;
        PSF(PSF<0) = 0 ;

        // --- Thresholding
        // PSF = PSF.*Sampling_grid; % Truncated PSF
        PSF = PSF % Sampling_grid;

        // PSF = PSF/sum(PSF(:));
        PSF = PSF / sum(PSF) ;
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
    arma::Mat<T> recon_matrix(const arma::Mat<T> Y, const arma::Mat<T> L, T Lambda)
    {
        using namespace arma;

        Mat<T> A, I, YY ;
        uword N;

        // YY = Y'*Y;
        YY = Y.t() * Y;

        // N = size(Y,1);
        //N = size(YY, 0);

        // I = diag(ones(N,1));
        //I = diagmat(ones<uvec>(N));

        // A = (YY + Lambda*L)\Y';
        A = solve (YY + Lambda * L, Y.t(), solve_opts::equilibrate);
        return A ;
    }
}

#endif

