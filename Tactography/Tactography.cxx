#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiffusionTensor3D.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkTensorFractionalAnisotropyImageFilter.h"
#include "itkVector.h"
#include "itkImageIterator.h"

//---Image Typedefs--//
const unsigned int nDims = 3 ;   // Setup types
typedef itk::DiffusionTensor3D < double > TensorType ;
typedef itk::Image < TensorType , nDims > ImageType ;
typedef itk::Vector <double, nDims> VectorType;
typedef itk::Image <VectorType, nDims> PAImageType ;
typedef itk::Image < double, nDims> BaseImageType ;
typedef itk::TensorImageFileReader < ImageType > TensorImageFileReaderType ;

int main ( int argc, char * argv[] )
{
  // --- Verify command line arguments----//
  if( argc < 7 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputImageFile inputSegmentedCCFile" <<
                              " outputPrincipalEigenVectorImageFile" <<
                              " outputFractionalAnisotropyImageFile" <<
                              " outputSeedTrackImageFile outputSegmentTrackImageFile" << std::endl ; 
      return -1 ;
    }

  
  //---Read Input Images---//
  TensorImageFileReaderType ::Pointer inputFileReader = TensorImageFileReaderType ::New() ;  
  inputFileReader->SetFileName ( argv[1] ) ;
  inputFileReader->Update();   // go read
  ImageType::Pointer img = inputFileReader->GetOutput();


  //---(2) Compute principal eigenvector---//
  ImageType::RegionType newRegion;
  ImageType::SizeType size = img->GetLargestPossibleRegion().GetSize() ;
  ImageType::IndexType starter ;
  starter[0] = 0; starter[1] = 0; starter[2] = 0;

  newRegion.SetSize( size ) ;
  newRegion.SetIndex( starter ) ;
  typedef itk::ImageRegionIterator < ImageType > InputImageIterator ;
  typedef itk::ImageRegionIterator < PAImageType > PAImageIterator ;

  PAImageType::Pointer paImage = PAImageType::New() ;
  paImage->SetOrigin(img->GetOrigin() ) ;
  paImage->SetDirection(img->GetDirection() );
  paImage->SetSpacing(img->GetSpacing() );
  paImage->SetRegions(newRegion);
  paImage->Allocate() ;

  PAImageIterator paIterator (paImage, newRegion);
  InputImageIterator inputImageIterator (img, newRegion);
  paIterator.GoToBegin ();
  inputImageIterator.GoToBegin () ;
  TensorType thisTensor ;
  TensorType::EigenValuesArrayType eigenValArrayType;
  TensorType::EigenVectorsMatrixType eigenValMatrixType ;
  VectorType thisVector ;

  while (!paIterator.IsAtEnd() )
  {
   thisTensor =  inputImageIterator.Value() ;
   thisTensor.ComputeEigenAnalysis(eigenValArrayType, eigenValMatrixType) ;
   thisVector[0] = eigenValMatrixType[2][0]*1 ; 
   thisVector[1] = eigenValMatrixType[2][1]*1 ; 
   thisVector[2] = eigenValMatrixType[2][2]*1 ; // Principal axis vector

   if (eigenValMatrixType[2][2] == 1) { 
     thisVector[0] = 0; 
     thisVector[1] = 0; 
     thisVector[2] = 0; //Change zero tensor to 0 Principal Vector Direction
   }

   paIterator.Set(thisVector) ;
   ++paIterator ;
   ++inputImageIterator ;
  }

  // Done.
  return 0 ;
}

