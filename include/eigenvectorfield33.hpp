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

#ifndef EIGENVECTORFIELD33_H
#define EIGENVECTORFIELD33_H

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1


#include "common.hpp"

#include <plog/Log.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <armadillo>

namespace phardi {


    template <typename T>
    arma::Row<T> normm(const arma::Mat<T> M) {
        using namespace arma;

        // % This script computes the L2 Norm for
        // norma = sqrt(sum((M').^2))';

        return sqrt(sum(pow(M.t(),2)));
    }


/*
% Calculate the eigenvectos of massive 3x3 real symmetric matrices given its eigenvalues.

% [eigenvector1,eigenvector2,eigenvector3] = eigenvaluefield33( a11, a12, a13, a22, a23, a33, l1, l2, l3);
% a11, a12, a13, a22, a23 and a33 specify the symmetric 3x3 real symmetric
% matrices as:
%   a11 a12 a13
% [ a12 a22 a13 ]
%   a13 a23 a33
*/
    template <typename T>
    void eigenvectorfield33(const arma::Row<T>  a11,
                            const arma::Row<T>  a12,
                            const arma::Row<T>  a13,
                            const arma::Row<T>  a22,
                            const arma::Row<T>  a23,
                            const arma::Row<T>  a33,
                            const arma::Row<T>  l1,
                            const arma::Row<T>  l2,
                            const arma::Row<T>  l3,
                            std::vector<arma::Row<T>> &e1,
                            std::vector<arma::Row<T>> &e2,
                            std::vector<arma::Row<T>> &e3) {

        using namespace arma;

        Row<T> A, B , C;
        Row<T> normal;

        // % eigenvector1
        // A = a11 - l1;
        A = a11 - l1;

        // B = a22 - l1;
        B = a22 - l1;

        // C = a33 - l1;
        C = a33 - l1;

        // e1x = (a12.*a23-B.*a13).*(a13.*a23-C.*a12);
        Row<T> e1x = (a12 % a23 - B % a13) % (a13 % a23 - C % a12);

        // e1y = (a13.*a23-C.*a12).*(a13.*a12-A.*a23);
        Row<T> e1y = (a13 % a23 - C % a12) % (a13 % a12 - A % a23);

        // e1z = (a12.*a23-B.*a13).*(a13.*a12-A.*a23);
        Row<T> e1z = (a12 % a23 - B % a13) % (a13 % a12 - A % a23);

        // normal = normm([e1x; e1y; e1z]');
        normal = normm<T>(join_vert(e1x,join_vert(e1y,e1z)).t() );

        // e1x = e1x(:)./normal;
        e1x = e1x / normal;
        e1.push_back(e1x);

        // e1y = e1y(:)./normal;
        e1y = e1y / normal;
        e1.push_back(e1y);

        // e1z = e1z(:)./normal;
        e1z = e1z / normal;
        e1.push_back(e1z);

        // % eigenvector2
        // A = a11 - l2;
        A = a11 - l2;

        // B = a22 - l2;
        B = a22 - l2;

        // C = a33 - l2;
        C = a33 - l2;

        // e2x = (a12.*a23-B.*a13).*(a13.*a23-C.*a12);
        Row<T> e2x = (a12 % a23 - B % a13) % (a13 % a23 - C % a12);

        // e2y = (a13.*a23-C.*a12).*(a13.*a12-A.*a23);
        Row<T> e2y = (a13 % a23 - C % a12) % (a13 % a12 - A % a23);

        // e2z = (a12.*a23-B.*a13).*(a13.*a12-A.*a23);
        Row<T> e2z = (a12 % a23 - B % a13) % (a13 % a12 - A % a23);

        // normal = normm([e2x; e2y; e2z]');
        normal = normm<T>( join_vert(e2x,join_vert(e2y,e2z)).t() );

        // e2x = e2x(:)./normal;
        e2x = e2x / normal;
        e2.push_back(e2x);

        // e2y = e2y(:)./normal;
        e2y = e2y / normal;
        e2.push_back(e2y);

        // e2z = e2z(:)./normal;
        e2z = e2z / normal;
        e2.push_back(e2z);

        // % eigenvector2
        // A = a11 - l3;
        A = a11 - l3;

        // B = a22 - l3;
        B = a22 - l3;

        // C = a33 - l3;
        C = a33 - l3;

        // e3x = (a12.*a23-B.*a13).*(a13.*a23-C.*a12);
        Row<T> e3x = (a12 % a23 - B % a13) % (a13 % a23 - C % a12);

        // e3y = (a13.*a23-C.*a12).*(a13.*a12-A.*a23);
        Row<T> e3y = (a13 % a23 - C % a12) % (a13 % a12 - A % a23);

        // e3z = (a12.*a23-B.*a13).*(a13.*a12-A.*a23);
        Row<T> e3z = (a12 % a23 - B % a13) % (a13 % a12 - A % a23);

        // normal = normm([e3x; e3y; e3z]');
        normal = normm<T>( join_vert(e3x,join_vert(e3y,e3z)).t() );

        // e3x = e3x(:)./normal;
        e3x = e3x / normal;
        e3.push_back(e3x);

        // e3y = e3y(:)./normal;
        e3y = e3y / normal;
        e3.push_back(e3y);

        //e3z = e3z(:)./normal;
        e3z = e3z / normal;
        e3.push_back(e3z);

        return;
    }
}

#endif
