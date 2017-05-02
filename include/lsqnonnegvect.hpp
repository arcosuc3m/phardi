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

#ifndef LSQNONNEGVECT_H
#define LSQNONNEGVECT_H


#include <armadillo>

namespace phardi {

    // Based on https://github.com/libigl/libigl/blob/master/include/igl/sortrows.cpp
    void sortrows(
            const arma::umat & X,
            const bool ascending,
            arma::umat & Y,
            arma::uvec & IX)
    {
        using namespace std;
        using namespace arma;

        // Resize output
        const uword num_rows = X.n_rows;
        const uword num_cols = X.n_cols;

        Y.resize(num_rows,num_cols);

        IX.resize(num_rows);

        for(uword i = 0; i<num_rows; ++i)
        {
            IX(i) = i;
        }

        if (ascending) {
            auto index_less_than = [&X, num_cols](uword i, uword j) {
                for (uword c=0; c<num_cols; ++c) {
                    if (X(i, c) < X(j, c)) return true;
                    else if (X(j,c) < X(i,c)) return false;
                }
                return false;
            };
            std::sort(
                    IX.memptr(),
                    IX.memptr()+IX.n_elem,
                    index_less_than
            );
        } else {
            auto index_greater_than = [&X, num_cols](uword i, uword j) {
                for (uword c=0; c<num_cols; ++c) {
                    if (X(i, c) > X(j, c)) return true;
                    else if (X(j,c) > X(i,c)) return false;
                }
                return false;
            };
            std::sort(
                    IX.memptr(),
                    IX.memptr()+IX.n_elem,
                    index_greater_than
            );
        }
        for (uword j=0; j<num_cols; ++j) {
            for(uword i = 0; i<num_rows; ++i)
            {
                Y(i,j) = X(IX(i), j);
            }
        }
    }


    // %--------------------------------------------------------------------------
    // % Find unique columns of a matrix
    arma::umat uniqueCols (const arma::umat & A) {

        using namespace arma;

        // [srtX, srtIdx] = sortrows(A');
        uvec srtIdx;
        umat srtX;
        sortrows(A.t(), true, srtX, srtIdx);

        // dX = diff(srtX, 1, 1);
        imat dX = diff(conv_to<imat>::from(srtX), 1, 0);

        // unqIdx = [true; any(dX, 2)];
        uvec unqIdx = join_cols(ones<uvec>(1), any(dX, 1));

        // uniqueCols = A(:,srtIdx(unqIdx));
        umat uniqueCols = A.cols(srtIdx(find(unqIdx)));

        return uniqueCols;
    }

/*
    %LSQNONNEGVECT Partly vectorized linear least squares with nonnegativity constraints based on
    %   MATLAB function lsqnonneg.
    %
    %   X = LSQNONNEGVECT(C,d) returns the matrix X, of which the kth column minimizes
    %   NORM(d(:,k)-C*X(:,k)) subject to X >= 0. C and d must be real. Each column of d corresponds
    %   to a distinct linear least squares problem
    %
    %   X = LSQNONNEGVECT(C,d,OPTIONS) minimizes with the default optimization
    %   parameters replaced by values in the structure OPTIONS, an argument
    %   created with the OPTIMSET function.  See OPTIMSET for details. Used
    %   options are Display and TolX. (A default tolerance TolX of
    %   10*MAX(SIZE(C))*NORM(C,1)*EPS is used.)
    %
    %   X = LSQNONNEGVECT(PROBLEM) finds the minimum for PROBLEM. PROBLEM is a
    %   structure with the matrix 'C' in PROBLEM.C, the vector 'd' in
    %   PROBLEM.d, the options structure in PROBLEM.options, and solver name
    %   'lsqnonneg' in PROBLEM.solver. The structure PROBLEM must have all the
    %   fields.
    %
    %   [X,RESNORM] = LSQNONNEGVECT(...) also returns the value of the squared 2-norm of
    %   the residual: norm(d-C*X)^2.
    %
    %   [X,RESNORM,RESIDUAL] = LSQNONNEGVECT(...) also returns the value of the
    %   residual: d-C*X.
    %
    %   [X,RESNORM,RESIDUAL,EXITFLAG] = LSQNONNEGVECT(...) returns an EXITFLAG that
    %   describes the exit condition of LSQNONNEGVECT. Possible values of EXITFLAG and
    %   the corresponding exit conditions are
    %
    %    1  LSQNONNEGVECT converged with a solution X.
    %    0  Iteration count was exceeded. Increasing the tolerance
    %       (OPTIONS.TolX) may lead to a solution.
    %
    %   [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT] = LSQNONNEGVECT(...) returns a structure
    %   OUTPUT with the number of steps taken in OUTPUT.iterations, the type of
    %   algorithm used in OUTPUT.algorithm, and the exit message in OUTPUT.message.
    %
    %   [X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT,LAMBDA] = LSQNONNEGVECT(...) returns
    %   the dual vector LAMBDA  where LAMBDA(i) <= 0 when X(i) is (approximately) 0
    %   and LAMBDA(i) is (approximately) 0 when X(i) > 0.
    %
    %   See also LSQNONNEG.

    %   version 2, 2014-08-06

    %   Adapted by David Provencher (Université de Sherbrooke, d.provencher@usherbrooke.ca)
    %   from the following version of Matlab's lsqnonneg function :
    %   $Revision: 1.15.4.14 $  $Date: 2009/11/16 22:27:07 $

    % Reference:
    %  Lawson and Hanson, "Solving Least Squares Problems", Prentice-Hall, 1974.
*/
    template <typename T>
    //function [x,resnorm,resid,exitflag,output,lambda] = lsqnonnegvect(C,d, options, varargin)
    arma::Mat<T> lsqnonnegvect (const arma::Mat<T> & C, const arma::Mat<T> & d) {
        using namespace arma;

        Col<T> resnorm;
        Mat<T> lambda;

        // tol = 10*eps*norm(C,1)*length(C);
        T tol = 10.0 * std::numeric_limits<T>::epsilon() * norm(C) * C.n_rows;

        // nModel = size(C,2);
        uword nModel = C.n_cols;

        //nData = size(d,2);
        uword nData = d.n_cols;

        // Initialize vector of nModel zeros and Infs (to be used later)
        //nZeros = zeros(nModel,nData);
        Mat<T> nZeros = zeros<Mat<T>>(nModel, nData);

        // % initilaize flag to indicate variables for which optimization is complete
        //outerOptimDone = false(1,nData);
        uvec outerOptimDone = zeros<uvec>(nData);

        //dataVect = 1:nData;
        uvec dataVect = regspace<uvec>(0,nData-1);

        //% Initialize set of non-active columns to null
        //P = false(nModel,nData);
        umat P = zeros<umat>(nModel, nData);

        //% Initialize set of active columns to all and the initial point to zeros
        //Z = true(nModel,nData);
        umat Z = ones<umat>(nModel, nData);

        //x = nZeros;
        Mat<T> x = nZeros;

        //resid = d - C*x;
        Mat<T> resid = d - C * x;
        //w = C' * resid;
        Mat<T> w = C.t() * resid;

        // % Set up iteration criterion
        //outerIter = 0;
        uword outerIter = 0;

        //innerIter = 0;
        uword innerIter = 0;

        //itmax = 3*nModel;
        uword itmax = 3 * nModel;

        //exitflag = 1;
        uword exitflag = 1;

        // % Data vectors with only zero or negative values do not need to be fitted
        //zeroFitMask = sum(abs(w),1) <= tol;
        uvec zeroFitMask = sum(abs(w),0).t() <= tol;


        //dataVectLeft = dataVect(~zeroFitMask);
        uvec dataVectLeft = dataVect(find(zeroFitMask == 0));

        //nDataLeft = numel(dataVectLeft);
        uword nDataLeft = dataVectLeft.n_elem;

        uvec t;
        uvec ind;

        //% Outer loop to put variables into set to hold positive coefficients
        //while nDataLeft > 0 % any(~outerOptimDone)
        while (nDataLeft > 0) {
            // outerIter = outerIter + 1;
            outerIter++;

            // % Reset intermediate solution z
            //z = zeros(nModel,nDataLeft);
            Mat<T> z = zeros<Mat<T>>(nModel,nDataLeft);

            // % Create wz, a Lagrange multiplier vector of variables in the zero set.
            // % wz must have the same size as w to preserve the correct indices, so
            // % set multipliers to -Inf for variables outside of the zero set.
            // wz = -Inf*ones(nModel, nDataLeft);
            Mat<T> wz = -datum::inf * ones<Mat<T>>(nModel, nDataLeft);

            // wz(Z(:,dataVectLeft)) = w(Z(:,dataVectLeft));
            wz(find(Z.cols(dataVectLeft))) = w(find(Z.cols(dataVectLeft)));

            // % Find variable for which optimisation is not done with largest Lagrange
            // % multiplier for each data vector
            // [~,t] = max(wz);
            t = index_max(wz).t();

            // % Faster version of sub2ind(size(wz), t(~outerOptimDone), dataVect(~outerOptimDone));
            // ind = t + (dataVectLeft-1)*nModel;
            ind = t + dataVectLeft * nModel;

            // % Move variable t from zero set to positive set
            //P(ind) = true;
            P(ind).ones();

            // Z(ind) = false;
            Z(ind).zeros();

            // % Compute intermediate solution using only variables in positive set
            //Punique = uniqueCols(P(:,dataVectLeft));
            umat Punique = uniqueCols(P.cols(dataVectLeft));

            // for k = 1:size(Punique,2)
            //     modelInd = Punique(:,k);
            //     colInd = find( all(bsxfun(@eq,P(:,dataVectLeft), modelInd), 1));
            //     globalInd = dataVectLeft(colInd);
            //     z(modelInd,colInd) = C(:,modelInd)\d(:,globalInd);
            // end

            for (uword k = 0; k < Punique.n_cols; ++k) {
                uvec modelInd = Punique.col(k);

                umat bsxfun(P.n_rows, dataVectLeft.n_elem);
                for (uword i = 0; i < dataVectLeft.n_elem; ++i) {
                    bsxfun.col(dataVectLeft(i)) = P.col(dataVectLeft(i)) == modelInd;
                }
                uvec colInd = find(all(bsxfun, 0));
                uvec globalInd = dataVectLeft(colInd);

                z(find(modelInd), colInd) = solve(C.cols(find(modelInd)), d.cols(globalInd));
            }

            // innerOptimDone = outerOptimDone;
            uvec innerOptimDone = outerOptimDone;

            // % inner loop to remove elements from the positive set which no longer
            // % belong
            //while any(z(P(:,dataVectLeft)) <= tol)
            while (any(z(find(P.cols(dataVectLeft))) <= tol) == true) {
                // innerIter = innerIter + 1;
                innerIter++;

                //if innerIter > itmax
                if (innerIter > itmax) {
                    LOG_INFO << "Exiting: Iteration count is exceeded, exiting LSQNONNEGVECT";
                    LOG_INFO << "Try raising the tolerance (OPTIONS.TolX).";

                    // output.iterations = outerIter;
                    // output.message = msg;
                    // output.algorithm = 'active-set';
                    // resnorm = sum(resid.*resid);
                    resnorm = sum(resid * resid);

                    // x = z;
                    x = z;

                    //lambda = w;
                    lambda = w;

                    return x;
                }
                // % Find indices where intermediate solution z is approximately negative
                // Q = (z <= tol) & P(:,dataVectLeft);
                umat Q = (z <= tol) && P.cols(dataVectLeft);


                // % This is equivalent to the lsqnonneg line alpha = min(x(Q)./(x(Q) - z(Q)))
                // % since Q here can have multiple columns
                // % Although a bit obscure, it can be 100-1000x faster than doing it in a loop

                // [~,indx] = find(Q);
                uvec indx = find (Q,1);

                // alpha = NaN*ones(1,nDataLeft);
                Col<T> alpha = datum::nan * ones<Col<T>>(nDataLeft);

                // ind = unique(indx);
                uvec ind  = find_unique(indx);

                // ind2 = dataVectLeft(ind);
                uvec ind2 =  dataVectLeft(ind);

                // alpha(ind) = min(x(:,ind2).*Q(:,ind) ./ (x(:,ind2).*Q(:,ind) - z(:,ind).*Q(:,ind)), [],1);
                alpha(ind) = min(x.cols(ind2) * Q.cols(ind) / x.cols(ind2) * Q.cols(ind) - z.cols(ind) * Q.cols(ind), 0);

                //ind = isnan(alpha);
                ind = find(alpha == datum::nan);

                // innerOptimDone(dataVectLeft(ind)) = true;
                innerOptimDone.elem(dataVectLeft.elem(ind > 0)).ones();

                // x(:,dataVectLeft(ind)) = z(:,ind);
                x.cols(dataVectLeft(ind)) = z.cols(ind);

                // x(:,dataVectLeft(~ind)) = x(:,dataVectLeft(~ind)) + bsxfun(@times, (z(:,~ind) - x(:,dataVectLeft(~ind))), alpha(~ind));
                x.cols(dataVectLeft(find(ind == 0))) = x.cols(dataVectLeft.elem(find(ind == 0))) +  (z.cols(find(ind == 0)) - x.cols(dataVectLeft.elem(find(ind == 0))))   * alpha.elem(find(ind == 0));

                // dataVectLeft = dataVectLeft(~ind);
                dataVectLeft = dataVectLeft.elem (find (ind == 0));

                //nDataLeft = length(dataVectLeft);
                nDataLeft = dataVectLeft.n_rows;

                // % Reset Z and P given intermediate values of x
                // Z(:,dataVectLeft) = ((abs(x(:,dataVectLeft)) < tol) & P(:,dataVectLeft)) | Z(:,dataVectLeft);
                Z.cols(dataVectLeft) = ((abs(x.cols(dataVectLeft)) < tol) && P.cols(dataVectLeft)) || Z.cols(dataVectLeft);

                // P(:,dataVectLeft) = ~Z(:,dataVectLeft);
                P.cols(dataVectLeft) = Z.cols(dataVectLeft) == 0;

                // z = zeros(nModel,nDataLeft);
                z = zeros<Mat<T>>(nModel, nDataLeft);

                // % Re-solve for z in unfinished optimizations
                //Punique = uniqueCols(P(:,dataVectLeft));
                umat Punique = uniqueCols(P.cols(dataVectLeft));

                // for k = 1:size(Punique,2)
                //     modelInd = Punique(:,k);
                //     colInd = find( all(bsxfun(@eq,P(:,dataVectLeft), modelInd), 1));
                //     globalInd = dataVectLeft(colInd);
                //     z(modelInd,colInd) = C(:,modelInd)\d(:,globalInd);
                // end

                for (uword k = 0; k < Punique.n_cols; ++k) {
                    uvec modelInd = Punique.col(k);
                    umat bsxfun(P.n_rows, dataVectLeft.n_elem);
                    for (uword i = 0; i < dataVectLeft.n_elem; ++i) {
                        bsxfun.col(dataVectLeft(i)) = P.col(dataVectLeft(i)) == modelInd;
                    }

                    uvec colInd = find(all(bsxfun, 0));
                    uvec globalInd = dataVectLeft(colInd);

                    z(find(modelInd), colInd) = solve(C.cols(find(modelInd)), d.cols(globalInd));
                }

            }

            // x(:,dataVectLeft) = z(:,1:nDataLeft);
            x.cols(dataVectLeft) = z.cols(regspace<uvec>(0,nDataLeft-1));


            // % Slow
            // dataVectLeft = dataVect(~outerOptimDone);
            dataVectLeft = dataVect(find(outerOptimDone == 0));

            // nDataLeft = length(dataVectLeft);
            nDataLeft = dataVectLeft.n_elem;

            // resid = d(:,dataVectLeft) - C*x(:,dataVectLeft);
            resid = d.cols(dataVectLeft) - C * x.cols(dataVectLeft);

            // w = C'*resid;
            w = C.t() * resid;

            //doneFlag = false(1,nDataLeft);
            uvec doneFlag = zeros<uvec>(nDataLeft);

            // doneFlag(~any(Z(:,dataVectLeft),1) | ~any(w.*Z(:,dataVectLeft) > tol, 1)) = true;
            umat anyl = any(Z.cols(dataVectLeft), 0);
            umat anyr = any(w % Z.cols(dataVectLeft) > tol, 0);
            doneFlag( find(anyl == 0 || anyr == 0 )).ones();

            // outerOptimDone(dataVectLeft) = doneFlag;
            outerOptimDone(dataVectLeft) = doneFlag;

            // % Remove values in w which are no longer necessary
            // w = w(:,~doneFlag);
            w = w.cols(find(doneFlag == 0));

            // dataVectLeft = dataVectLeft(~doneFlag);
            dataVectLeft = dataVectLeft(find(doneFlag == 0));

            // nDataLeft = length(dataVectLeft);
            nDataLeft = dataVectLeft.n_elem;

            // innerIter = 0;
            innerIter = 0;
        }

        // % Recompute residual using all data vectors

        // resid = d - C*x;
        resid = d - C * x;

        // resnorm = sum(resid.^2);
        resnorm = sum(pow(resid,2)).t();

        //lambda = C'*resid;
        lambda = C.t() * resid;

        LOG_INFO << "Optimization terminated.";
        return x;
    }
}

#endif