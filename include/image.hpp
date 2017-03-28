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

#ifndef IMAGE_H
#define IMAGE_H

#include <itkImage.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include <itkOrientImageFilter.h>
#include <itkNiftiImageIO.h>
#include <itkGDCMImageIO.h>
#include <itkRGBAPixel.h>

namespace phardi {

    typedef float                       PixelType;
    typedef itk::Image< PixelType, 2 >  Image2DType;
    typedef itk::Image< PixelType, 3 >  Image3DType;
    typedef itk::Image< PixelType, 4 >  Image4DType;
    
    typedef itk::NiftiImageIO           NiftiType;
    typedef itk::GDCMImageIO            DicomType;

    typedef Image2DType::IndexType      Index2DType;
    typedef Image2DType::SizeType       Size2DType;
    typedef Image2DType::RegionType     Region2DType;
    typedef Image2DType::SpacingType    Spacing2DType;
    typedef Image2DType::PointType      Origin2DType;
    typedef Image2DType::DirectionType  Direction2DType;

    typedef Image3DType::IndexType      Index3DType;
    typedef Image3DType::SizeType       Size3DType;
    typedef Image3DType::RegionType     Region3DType;
    typedef Image3DType::SpacingType    Spacing3DType;
    typedef Image3DType::PointType      Origin3DType;
    typedef Image3DType::DirectionType  Direction3DType;
  
    typedef Image4DType::IndexType      Index4DType;
    typedef Image4DType::SizeType       Size4DType;
    typedef Image4DType::SpacingType    Spacing4DType;
    typedef Image4DType::PointType      Origin4DType;
    typedef Image4DType::DirectionType  Direction4DType;

    
    itk::ImageIOBase::Pointer getImageIO(std::string input){
        itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO(input.c_str(), itk::ImageIOFactory::ReadMode);

        imageIO->SetFileName(input);
        imageIO->ReadImageInformation();

        return imageIO;
    }

    itk::ImageIOBase::IOComponentType component_type(itk::ImageIOBase::Pointer imageIO){
        return imageIO->GetComponentType();
    }

    itk::ImageIOBase::IOPixelType pixel_type(itk::ImageIOBase::Pointer imageIO){
        return imageIO->GetPixelType();
    }

    size_t num_dimensions(itk::ImageIOBase::Pointer imageIO){
        return imageIO->GetNumberOfDimensions();
    }    

   template<typename TImageType>
   void ReadImage(std::string filename, typename TImageType::Pointer image)
   {
        typedef itk::ImageFileReader<TImageType> ReaderType;
        typename ReaderType::Pointer reader = ReaderType::New();
 
        reader->SetFileName(filename);
 
        try {
            reader->Update();
        } catch (itk::ExceptionObject &err) {
            LOG_ERROR << "ExceptionObject caught !";
            std::cerr <<  err << std::endl;
            std::cerr << "Error reading file " << filename << std::endl;
            
        }
        image->Graft(reader->GetOutput());
    }

    template<typename TImageType>
    void CreateImage(typename TImageType::Pointer image, 
                     typename TImageType::SizeType size, 
                     typename TImageType::IndexType index, 
                     typename TImageType::SpacingType spacing, 
                     typename TImageType::PointType origin,
                     typename TImageType::DirectionType direction){
        using namespace itk;

        typename TImageType::RegionType region;

        region.SetSize(size);
        region.SetIndex(index);
        image->SetRegions(region);
        image->SetSpacing(spacing);
        image->SetOrigin(origin);
        image->SetDirection(direction);
        image->Allocate();

    }

    template<typename TImageType, typename TImageIO>
    void WriteImage(std::string filename,  typename TImageType::Pointer image){
        using namespace itk;
        
        typename TImageIO::Pointer nifti_io = TImageIO::New();
        nifti_io->SetPixelType(itk::ImageIOBase::SCALAR);
        nifti_io->SetComponentType(itk::ImageIOBase::FLOAT);
       
        typename itk::ImageFileWriter<TImageType>::Pointer dwi_writer = itk::ImageFileWriter<TImageType>::New();
        dwi_writer->SetFileName(filename);
        dwi_writer->SetInput(image);
        dwi_writer->SetImageIO(nifti_io);
        try {
            dwi_writer->Update();
        } catch (itk::ExceptionObject &err) {
            LOG_ERROR << "ExceptionObject caught !";
            std::cerr << err << std::endl;            
            std::cerr << "Error writting file " << filename  << std::endl;
        }
    }
}

#endif

