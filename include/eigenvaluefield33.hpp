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

#ifndef EIGENVALUEFIELD33_H
#define EIGENVALUEFIELD33_H

#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1


#include "common.hpp"

#include <plog/Log.h>
#include <iostream>
#include <math.h>
#include <cmath>
#include <armadillo>

namespace phardi {

/*
% Calculate the eigenvalues of massive 3x3 real symmetric matrices.
% Computation is based on matrix operation and GPU computation is 
% supported.
% Syntax:
% [eigenvalue1,eigenvalue2,eigenvalue3] = eigenvaluefield33( a11, a12, a13, a22, a23, a33)
% a11, a12, a13, a22, a23 and a33 specify the symmetric 3x3 real symmetric
% matrices as:
%   a11 a12 a13
% [ a12 a22 a13 ]
%   a13 a23 a33
% These six inputs must have the same size. They can be 2D, 3D or any
% dimension. The outputs eigenvalue1, eigenvalue2 and eigenvalue3 will
% follow the size and dimension of these inputs. Owing to the use of 
% trigonometric functions, the inputs must be double to maintain the 
% accuracy.
%
% eigenvalue1, eigenvalue2 and eigenvalue3 are the unordered resultant 
% eigenvalues. They are solved using the cubic equation solver, see
% http://en.wikipedia.org/wiki/Eigenvalue_algorithm
%
% The peak memory consumption of the method is about 1.5 times of the total 
% of all inputs, in addition to the original inputs. GPU computation is 
% used automatically if all inputs are matlab GPU arrays. 
%
% Author: Max W.K. Law
% Email:  max.w.k.law@gmail.com
% Page:   http://www.cse.ust.hk/~maxlawwk/
*/
    template <typename T>
    void eigenvaluefield33(const arma::Col<T>  a11,
                           const arma::Col<T>  a12,
                           const arma::Col<T>  a13,
                           const arma::Col<T>  a22,
                           const arma::Col<T>  a23,
                           const arma::Col<T>  a33,
                           arma::Col<T> &b,
                           arma::Col<T> &j,
                           arma::Col<T> &d) {

        using namespace arma;

        //ep=1e-50;
        //b=double(a11)+ep;
        b = a11 + datum::eps;

        //d=double(a22)+ep;
        d = a22 + datum::eps;

        //j=double(a33)+ep;
        j = a33 + datum::eps;

        //c=-(double(a12).^2 + double(a13).^2 + double(a23).^2 - b.*d - d.*j - j.*b);
        Col<T> c =-( pow(a12,2) + pow(a13,2) + pow(a23,2) - b * d - d * j - j * b);
        
        //d=-(b.*d.*j - double(a23).^2.*b - double(a12).^2.*j - double(a13).^2.*d + 2*double(a13).*double(a12).*double(a23));
        d =-( b*d*j - pow(a23,2) * b - pow(a12,2) * j - pow(a13,2) * d + 2 * a13 * a12 * a23);

        //b=-double(a11) - double(a22) - double(a33) - ep - ep -ep;

        //d = d + ((2*b.^3) - (9*b.*c))/27;
        d = d + (pow(2*b,3) - (9.0f * b * c))/27.0f;

        //%%%c=(d.^2/4-(d.^2/4 + ((3 * c - (b.^2))/3).^3/27));
        //c=(b.^2)/3 - c;
        c = pow(b,2)/3.0f - c;

        //c=c.^3;
        c = pow(c,3);

        //c=c/27;
        c = c/27.0f;

        //c=max(c,0);
        c = max(c,0);

        //c=realsqrt(c);
        c = sqrt(c);    
    
        //j=c.^(1/3);
        j = pow(c,1.0f/3.0f);

        c=c+(c==0);

        //d=-d/2./c;
        d = -d/2/c;

        //d=min(d, 1);
        d = min(d, 1);

        //d=max(d, -1);
        d = max(d, -1);
    
        //d=real(acos(d)/3);
        d = acos(d)/3.0f;
    
        //c=j.*cos(d);
        c = j * cos(d);

        //d=j.*sqrt(3).*sin(d);
        d = j * sqrt(3.0) * sin(d);

        //b=-b/3;
        b = -b/3.0f;
    
        //j = single(-c-d+b);
        j = -c-d+b;
 
        //d = single(-c+d+b);
        d = -c+d+b;

        //b = single(2*c+b);
        b = 2*c+b;

        return;
    }
}

#endif
