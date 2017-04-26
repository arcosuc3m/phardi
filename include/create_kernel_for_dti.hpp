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

#ifndef CREATE_KERNEL_DTI_H
#define CREATE_KERNEL_DTI_H

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1


#include "common.hpp"

#include <plog/Log.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <armadillo>

namespace phardi {

/*
    %-fanDTasia ToolBox------------------------------------------------------------------
    % This Matlab script is part of the fanDTasia ToolBox: a Matlab library for Diffusion 
    % Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, High-order 
    % Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.
    %
    % A Matlab Tutorial on DW-MRI can be found in:
    % http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
    %
    %-CITATION---------------------------------------------------------------------------
    % If you use this software please cite the following work:
    % A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
    % of any order with Symmetric Positive-Definite Constraints", 
    % In the Proceedings of ISBI, 2010
    %
    %-DESCRIPTION------------------------------------------------------------------------
    % This function computes all possible monomials in 3 variables of a certain order. 
    % The 3 variables are given in the form of 3-dimensional vectors.
    %
    %-USE--------------------------------------------------------------------------------
    % G=constructMatrixOfMonomials(g,order);
    %
    % g: is a list of N 3-dimensional vectors stacked in a matrix of size Nx3
    % order: is the order of the computed monomials
    % G: contains the computed monomials. It is a matrix of size N x (2+order)!/2(order)!
    %
    %-DISCLAIMER-------------------------------------------------------------------------
    % You can use this source code for non commercial research and educational purposes 
    % only without licensing fees and is provided without guarantee or warrantee expressed
    % or implied. You cannot repost this file without prior written permission from the 
    % authors. If you use this software please cite the following work:
    % A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
    % of any order with Symmetric Positive-Definite Constraints", In Proc. of ISBI, 2010.
    %
    %-AUTHOR-----------------------------------------------------------------------------
    % Angelos Barmpoutis, PhD
    % Computer and Information Science and Engineering Department
    % University of Florida, Gainesville, FL 32611, USA
    % abarmpou at cise dot ufl dot edu
    %------------------------------------------------------------------------------------
*/

    template <typename T>
    arma::Mat<T> constructMatrixOfMonomials(const arma::Mat<T> & g,
                                            const arma::uword order) {

        using namespace arma;

        // %fprintf(1,'Constructing matrix of monomials G...');
        // for k=1:length(g)
        //     c=1;
        //     for i=0:order
        //         for j=0:order-i
        //             G(k,c)=(g(k,1)^i)*(g(k,2)^j)*(g(k,3)^(order-i-j));
        //             c=c+1;
        //         end
        //     end
        // end

        Mat<T> G(g.n_rows, factorial(order+1));
        G.zeros();

        uword c = 0;
        for (uword k = 0; k < g.n_rows; ++k) {
            c = 0;
            for (uword i = 0; i <= order; ++i) {
                for (uword j = 0; j <= order-i; ++j) {
                    G(k,c) = (pow(g(k,0),i)) * (pow(g(k,1),j)) * (pow(g(k,2),order-i-j));
                    c++;
                }
            }
        }

        return G;
    }

/*
    %-fanDTasia ToolBox------------------------------------------------------------------
    % This Matlab script is part of the fanDTasia ToolBox: a Matlab library for Diffusion 
    % Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, High-order 
    % Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.
    %
    % A Matlab Tutorial on DW-MRI can be found in:
    % http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
    %
    %-CITATION---------------------------------------------------------------------------
    % If you use this software please cite the following work:
    % A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
    % of any order with Symmetric Positive-Definite Constraints", 
    % In the Proceedings of ISBI, 2010
    %
    %-DESCRIPTION------------------------------------------------------------------------
    % This function computes how many identical copies of a tensor coefficient appear in
    % a fully symmetric high order tensor in 3 variables of spesific order.
    %
    %-USE--------------------------------------------------------------------------------
    % counter=population(i,j,k,order)
    %
    % order: is the order of the symmetric tensor in 3 variables (x,y,z)
    % i,j,k: is a triplet that represent the coefficient which is weighted by the monomial
    % x^i*y^j*z^k, where i+j+k=order
    %
    %-DISCLAIMER-------------------------------------------------------------------------
    % You can use this source code for non commercial research and educational purposes 
    % only without licensing fees and is provided without guarantee or warrantee expressed
    % or implied. You cannot repost this file without prior written permission from the 
    % authors. If you use this software please cite the following work:
    % A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
    % of any order with Symmetric Positive-Definite Constraints", In Proc. of ISBI, 2010.
    %
    %-AUTHOR-----------------------------------------------------------------------------
    % Angelos Barmpoutis, PhD
    % Computer and Information Science and Engineering Department
    % University of Florida, Gainesville, FL 32611, USA
    % abarmpou at cise dot ufl dot edu
    %------------------------------------------------------------------------------------
*/

    arma::uvec populationBasis(const arma::uword i, const arma::uword order, arma::uvec c) {

        using namespace arma;

        // if order==0
        //     ret=c;
        // else
        //     j=mod(i,3);
        //     c(j+1)=c(j+1)+1;
        //     ret=populationBasis((i-j)/3,order-1,c);
        // end

        if (order == 0) {
            return c;
        } else {
            uword j = i % 3;
            c(j) = c(j) + 1;
            return populationBasis((i-j)/3, order-1, c);
        }
    }

    arma::uword population(const arma::uword i, const arma::uword j, const arma::uword k,
                    const arma::uword order) {
    
        using namespace arma;

        // size=3^order;
        // counter=0;
        // for z=0:size-1
        //     c=populationBasis(z,order,[0 0 0]);
        //     if (c(1)==i)&(c(2)==j)&(c(3)==k)
        //        counter=counter+1;
        //     end
        // end

        uword counter = 0;
        uword size = pow(3,order);
        uvec c(3);
        for (uword z = 0; z <= size -1; ++z) {
            c = populationBasis(z, order, zeros<uvec>(3));

            if (c(0) == i && c(1) == j && c(2) == k)
                counter++; 
        }

        return counter;

    }

/*
    %-fanDTasia ToolBox------------------------------------------------------------------
    % This Matlab script is part of the fanDTasia ToolBox: a Matlab library for Diffusion 
    % Weighted MRI (DW-MRI) Processing, Diffusion Tensor (DTI) Estimation, High-order 
    % Diffusion Tensor Analysis, Tensor ODF estimation, Visualization and more.
    %
    % A Matlab Tutorial on DW-MRI can be found in:
    % http://www.cise.ufl.edu/~abarmpou/lab/fanDTasia/tutorial.php
    %
    %-CITATION---------------------------------------------------------------------------
    % If you use this software please cite the following work:
    % A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
    % of any order with Symmetric Positive-Definite Constraints", 
    % In the Proceedings of ISBI, 2010
    %
    %-DESCRIPTION------------------------------------------------------------------------
    % This function computes the coefficients of homogenous polynomials in 3 variables of
    % a given order, which correspond to squares of lower (half) order polynomials. 
    %
    %-USE--------------------------------------------------------------------------------
    % C=constructSetOfPolynomialCoef(order);
    %
    % order: is the order of the computed homogeneous polynomials
    % C: contains the computed list of polynomial coefficients. 
    %    It is a matrix of size Mprime x (2+order)!/2(order)!
    %    Mprime is defined here =321. This implementation is one way of choosing polynomials,
    %    there are many different ways. A slightly different way is described in the above
    %    reference (Barmpoutis et al. ISBI'10).
    %
    %-DISCLAIMER-------------------------------------------------------------------------
    % You can use this source code for non commercial research and educational purposes 
    % only without licensing fees and is provided without guarantee or warrantee expressed
    % or implied. You cannot repost this file without prior written permission from the 
    % authors. If you use this software please cite the following work:
    % A. Barmpoutis and B.C. Vemuri, "A Unified Framework for Estimating Diffusion Tensors 
    % of any order with Symmetric Positive-Definite Constraints", In Proc. of ISBI, 2010.
    %
    %-AUTHOR-----------------------------------------------------------------------------
    % Angelos Barmpoutis, PhD
    % Computer and Information Science and Engineering Department
    % University of Florida, Gainesville, FL 32611, USA
    % abarmpou at cise dot ufl dot edu
    %------------------------------------------------------------------------------------
*/

    template <typename T>
    void create_Kernel_for_dti(const arma::Mat<T> & V,
                                 const arma::Mat<T> & diffGrads,
                                 const arma::Col<T> & diffBvals,
                                 arma::Mat<T> & Kernel,
                                 const phardi::options opts) {

        using namespace arma;

        // for i=0:order
        //     for j=0:order-i
        //         pop(i+1,j+1,order-i-j+1)=population(i,j,order-i-j,order);
        //     end
        // end

        uword order = opts.dti_nnls.torder;
   
        ucube pop(order+1,order+1,order+1);
        pop.zeros();

        for (uword i = 0; i <= order; ++i) {
            for (uword j = 0; j <= order - i; ++j) {
                pop(i, j, order-i-j) = population(i, j, order-i-j, order);
            } 
        }

        // for k=1:length(g)
        //     c=1;
        //     for i=0:order
	//         for j=0:order-i
        //             C(k,c)=pop(i+1,j+1,order-i-j+1)*(g(k,1)^i)*(g(k,2)^j)*(g(k,3)^(order-i-j));
        //             c=c+1;
        //         end
        //     end
        // end

        Mat<T> C (V.n_rows, factorial(order+1));
        C.zeros();

        uword c = 0;
        for (uword k = 0; k < V.n_rows; ++k) {
            c = 0;
            for (uword i = 0; i <= order; ++i) {
                for (uword j = 0; j <=  order - i; ++j) {
                    C(k,c) = pop(i, j, order-i-j) * (pow(V(k,0), i)) * (pow(V(k,1),j)) * (pow(V(k,2),order-i-j));
                    c++;
                }
            }    
        }

        // G=constructMatrixOfMonomials(diffGrads, 2);
        Mat<T> G = constructMatrixOfMonomials(diffGrads, 2);

        // Kernel=G*C';
        Kernel = G * C.t();

        // Kernel=[-diag(diffBvals)*Kernel ones(size(diffGrads,1),1)];
        Kernel = join_cols(-diagmat(diffBvals) * Kernel, ones<Col<T>>(diffBvals.n_rows));
        return;
    }
}

#endif
