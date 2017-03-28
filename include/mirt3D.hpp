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

#ifndef MIRT3D_H
#define MIRT3D_H

#include <armadillo>

namespace phardi {



    template <typename Te>
    void mirt3D(
            const arma::Cube<Te> & Z,
            const arma::Mat<Te> & S,
            const arma::Mat<Te> & T,
            const arma::Mat<Te> & W,
            arma::Mat<Te> & F,
            arma::uword	MN,
            arma::uword nrows,
            arma::uword ncols,
            arma::uword npages,
            arma::uword ndim
    )
    {
        using namespace arma;

        uword n, in1, in2, in3, in4, in5, in6, in7, in8;
        Te t, s,  s1, w, w1, tmp, nan;
        Te m1, m2, m3, m4, m5, m6, m7, m8;
        uword ndx, nw, Zshift, i, nrowsncols, ft, fs, fw;

        nw         = nrows*ncols;
        nrowsncols = nw*npages;
        nan        = datum::nan;

        for (n=0; n < MN; ++n) {

            t=T(n);
            s=S(n);
            w=W(n);

            ft=(uword) floor(t);
            fs=(uword) floor(s);
            fw=(uword) floor(w);


            if (fs<1 || s>ncols || ft<1 || t>nrows || fw<1 || w>npages){
                /* Put nans if outside*/
                for (i = 0; i < ndim; i++) F(n+i*MN) = nan;
            }
            else  {

                ndx =  ft+(fs-1)*nrows+(fw-1)*nw;

                if (s==ncols){ s=s+1; ndx=ndx-nrows; }
                s=s-fs;
                if (t==nrows){  t=t+1; ndx=ndx-1; }
                t=t-ft;
                if (w==npages){  w=w+1; ndx=ndx-nw; }
                w=w-fw;

                in1=ndx-1;
                in2=ndx;
                // in3=ndx+nrows-1;
                in3=in1+nrows;
                // in4=ndx+nrows;
                in4=in3+1;

                // in5=ndx+nw-1;
                in5=in1+nw;
                // in6=ndx+nw;
                in6=in5+1;
                // in7=ndx+nrows+nw-1;
                in7=in5+nrows;
                // in8=ndx+nrows+nw;
                in8 = in7+1;

                ////////////
                for (i = 0; i < ndim; ++i){

                    s1=1-s;
                    w1=1-w;

                    tmp=s1*w1;
                    m2=t*tmp;
                    m1=tmp-m2;

                    tmp=s*w1;
                    m4=t*tmp;
                    m3=tmp-m4;

                    tmp=s1*w;
                    m6=t*tmp;
                    m5=tmp-m6;

                    tmp=s*w;
                    m8=t*tmp;
                    m7=tmp-m8;

                    Zshift=i*nrowsncols;
                    F(n+i*MN)=Z(in1+Zshift)*m1+Z(in2+Zshift)*m2+Z(in3+Zshift)*m3+Z(in4+Zshift)*m4+Z(in5+Zshift)*m5+Z(in6+Zshift)*m6+
                              Z(in7+Zshift)*m7+Z(in8+Zshift)*m8;

                }

            }

        } // cycle end

        return;
    }


    template <typename T>
    arma::Mat<T> mirt3D_Function(const arma::Cube<T> & in_z, const arma::Mat<T> &in_s, const arma::Mat<T> &in_t, const arma::Mat<T> &in_w)
    {

        using namespace arma;

        uword i, MN, nrows, ncols, npages, vol, ndim, newXndim, Xndim;
        uword dims[2], Xdims[2], newdims[20];

        /* Get the sizes of each input argument */

        Xndim = 2;
        Xdims[0] = in_s.n_rows;
        Xdims[1] = in_s.n_cols;

        ndim = 3;
        dims[0] = in_z.n_rows;
        dims[1] = in_z.n_cols;
        dims[2] = in_z.n_slices;

        MN = 1;
        /*Total number of interpolations points in 1 image*/
        for (i = 0; i < Xndim; ++i) {
            MN = MN*Xdims[i];
        }

        vol=1; newXndim=Xndim;
        if (ndim>3) {   /*Check if we have several images*/
            if ((Xndim==2) && (Xdims[1]==1))  {newXndim=newXndim-1; }  /*Check if interpolate along column vectors*/
            for (i = 0; i < newXndim; i++) {newdims[i]=Xdims[i];};  /*Copy original dimenstions*/
            newdims[newXndim]=dims[3];                             /*Add the number of  images as a last dimenstion*/
            newXndim=newXndim+1;                                       /*Set the new number of dimenstions*/
            vol=dims[3];
        }
        else
        {
            for (i = 0; i < newXndim; i++) {newdims[i]=Xdims[i];};
        }

        Mat<T> out_f (newdims[0], newdims[1]);

        /* Input image size */
        nrows = dims[0];
        ncols = dims[1];
        npages = dims[2];

        /* Do the actual computations in a subroutine */
        mirt3D(in_z, in_s, in_t, in_w, out_f, MN, nrows, ncols, npages, vol);

        return out_f;
    }
}

#endif

