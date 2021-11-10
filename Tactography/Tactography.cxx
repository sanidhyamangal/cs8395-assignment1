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

    // Done
    return 0
}
