#include "itkImage.h"
#include "itkDiffusionTensor3DReconstructionImageFilter.h"
#include "itkTensorFractionalAnisotropyImageFilter.h"
#include "itkTensorRelativeAnisotropyImageFilter.h"
#include "itkNrrdImageIO.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkImageRegionIterator.h"
#include "itkDiffusionTensor3D.h"
#include "itkImageRegionIterator.h"
#include "itkImageToVTKImageFilter.h"

int main ( int argc, char * argv[] )
{
  // Verify command line arguments
  if( argc < 3 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputImageFile outputImageFile" << std::endl ; 
      return -1 ;
    }

    // define all the datatype
    const unsigned int nDims = 3;

    // define tensor type and image type 
    typedef itk::DiffusionTensor3D <float> TensorType;
    typedef itk::Image <TensorType, nDims> ImageType;
    typedef itk::ImageFileReader <ImageType> ImageFileReader;

    ImageFileReader::Pointer imgReader = ImageFileReader::New();
    imgReader -> SetFileName(argv[1]);
    ImageType::Pointer img = imgReader -> GetOutput();
    ImageType::RegionType region = img -> GetLargestPossibleRegion();
    ImageType::SizeType size = region -> GetSize();

    std::cout << size;

    // Done
    return 0;
}
