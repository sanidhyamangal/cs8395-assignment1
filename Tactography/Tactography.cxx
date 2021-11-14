#include "itkImage.h"
#include "itkDiffusionTensor3DReconstructionImageFilter.h"
#include "itkTensorFractionalAnisotropyImageFilter.h"
#include "itkTensorRelativeAnisotropyImageFilter.h"
#include "itkNrrdImageIO.h"
#include "itkImageSeriesReader.h"
#include "itkImageFileWriter.h"
#include "itkDiffusionTensor3D.h"
#include "itkImageRegionIterator.h"
#include "itkImageToVTKImageFilter.h"
#include "itkImageFileReader.h"

int main ( int argc, char * argv[] )
{
  // Verify command line arguments
  if( argc < 6 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputImageFile inputSegmentFile outputEigenvectorFile outputFAFile outputSegmentTrackFile" << std::endl ; 
      return -1 ;
    }

    // define all the datatype
    const unsigned int nDims = 3;

    // define tensor type and image type 
    typedef itk::DiffusionTensor3D <double> TensorType;
    typedef itk::Image <TensorType, nDims> TensorImageType;
    typedef itk::Image <double, nDims> ImageType;
    typedef itk::ImageFileReader <TensorImageType> TensorImageFileReader;
    typedef itk::ImageFileReader <ImageType> ImageFileReader;
    typedef itk::Vector <double, nDims> VectorType;
    typedef itk::Image <VectorType, nDims> PAImageType;

    // define eigne value matrix and eigenvalue array along with vector and tensortype
    TensorType thisTensor;
    VectorType thisVector;
    TensorType::EigenValuesArrayType eigenValArrayType;
    TensorType::EigenVectorsMatrixType eigenValMatrixType;

    // define iters for the image and PAImageType
    typedef itk::ImageRegionIterator <TensorImageType> ImageIteratorType;
    typedef itk::ImageRegionIterator <PAImageType> PAImageIteratorType;


    // load input image for the processing
    TensorImageFileReader::Pointer imgReader = TensorImageFileReader::New();
    imgReader -> SetFileName(argv[1]);
    imgReader -> Update();
    TensorImageType::Pointer img = imgReader -> GetOutput();
    TensorImageType::RegionType region = img -> GetLargestPossibleRegion();
    TensorImageType::SizeType size = region.GetSize();


    // compute principal eigen value vector
    // create a new image type track of voxels

    TensorImageType::RegionType newRegion;
    //define new index type
    TensorImageType::IndexType origin;
    origin[0]=0;
    origin[1]=0;
    origin[2]=0;

    newRegion.SetSize(size);
    newRegion.SetIndex(origin);

    PAImageType::Pointer paImage = PAImageType::New();

    // define PAImage based on inputImage and new origin
    paImage -> SetOrigin(img->GetOrigin());
    paImage->SetDirection(img->GetDirection() );
    paImage->SetSpacing(img->GetSpacing() );
    paImage->SetRegions(newRegion);
    paImage->Allocate() ; // allocate this as a memory space


    // iterate through the input image and PA image
    PAImageIteratorType paImageIterator (paImage, newRegion);
    ImageIteratorType inputImageIterator(img, newRegion);


  typedef itk::ImageRegionIterator < ImageType > InputIteratorType ;
  typedef itk::ImageRegionIterator < PAImageType > OutputIteratorType ;

  PAImageType::Pointer myPAImage = PAImageType::New() ;
  myPAImage->SetOrigin(myImage->GetOrigin() ) ;
  myPAImage->SetDirection(myImage->GetDirection() );
  myPAImage->SetSpacing(myImage->GetSpacing() );
  myPAImage->SetRegions(newRegion);
  myPAImage->Allocate() ;

  OutputIteratorType outputIterator (myPAImage, newRegion);
  InputIteratorType inputIterator (img, newRegion);
  outputIterator.GoToBegin ();
  inputIterator.GoToBegin () ;
  DiffTensorType thisTensor ;
  DiffTensorType::EigenValuesArrayType myEVAT;
  DiffTensorType::EigenVectorsMatrixType myEVMT ;
  VectorType thisVector ;

  while (!outputIterator.IsAtEnd() )
  {
   thisTensor =  inputIterator.Value() ;
   thisTensor.ComputeEigenAnalysis(myEVAT, myEVMT) ;
   thisVector[0] = myEVMT[2][0]*1 ; thisVector[1] = myEVMT[2][1]*1 ; thisVector[2] = myEVMT[2][2]*1 ; // Principal axis vector

   if (myEVMT[2][2] == 1) { 
     thisVector[0] = 0; thisVector[1] = 0; thisVector[2] = 0; //Change zero tensor to 0 Principal Vector Direction
   }
   outputIterator.Set(thisVector) ;
   ++outputIterator ;
   ++inputIterator ;
  }

    // // go to begin of the image iterate
    // paImageIterator.GoToBegin();
    // inputImageIterator.GoToBegin();
    
    // // compute eigenvals and eigne vectors for the image
    // while (!paImageIterator.IsAtEnd())
    // {
    //   thisTensor = inputImageIterator.Value();

    //   thisTensor.ComputeEigenAnalysis(eigenValArrayType, eigenValMatrixType);

    //   // assign eigen val and vector to the tensor
    //   thisVector[0]=eigenValMatrixType[2][0] * 1;
    //   thisVector[1]=eigenValMatrixType[2][1] * 1;
    //   thisVector[2]=eigenValMatrixType[2][2] * 1;

    //   // clip the vector if its zero i.e. matrixval[2][2] = 1
    //   if (eigenValMatrixType[2][2] == 1)
    //   {
    //     thisVector[0]=0;
    //     thisVector[1]=0;
    //     thisVector[2]=0;
    //   }

    //   // update iterator value to this vector direction
    //   paImageIterator.Set(thisVector)

    //   // increment iterators
    //   ++paImageIterator;
    //   ++inputImageIterator;
    // }
    

    // Done
    return 0;
}
