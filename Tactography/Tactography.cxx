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
    typedef itk::ImageFileReader <ImageType> ImageFileReader;
    typedef itk::Vector <double, nDims> VectorType;
    typedef itk::Image <VectorType, nDims> PAImageType;

    // define eigne value matrix and eigenvalue array along with vector and tensortype
    TensorType thisTensor;
    VectorType thisVector;
    TensorType::EigenValuesArrayType eigenValArrayType;
    TensorType::EigenVectorsMatrixType eigenValMatrixType;


    // load input image for the processing
    ImageFileReader::Pointer imgReader = ImageFileReader::New();
    inputFile -> SetFileName(argv[1]);
    TensorImageType::Pointer img = inputFile -> GetOutput();
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


    // define iters for the image and PAImageType
    typedef itk::ImageRegionIterator <TensorImageType> ImageIteratorType;
    typedef itk::ImageRegionIterator <PAImageType> PAImageIteratorType;

    // iterate through the input image and PA image
    PAImageIteratorType paImageIterator (paImage, newRegion);
    ImageIteratorType inputImageIterator(img, newRegion);

    // go to begin of the image iterate
    paImageIterator.GoToBegin();
    inputImageIterator.GoToBegin();
    
    // compute eigenvals and eigne vectors for the image
    while (!paImageIterator.IsAtEnd())
    {
      thisTensor = inputImageIterator.Value();

      thisTensor.ComputeEgienAnalysis(eigenValArrayType, eigenValMatrixType);

      // assign eigen val and vector to the tensor
      thisVector[0]=eigneValMatrixType[2][0] * 1;
      thisVector[1]=eigneValMatrixType[2][1] * 1;
      thisVector[2]=eigneValMatrixType[2][2] * 1;

      // clip the vector if its zero i.e. matrixval[2][2] = 1
      if (eigenValMatrixType[2][2] == 1)
      {
        thisVector[0]=0;
        thisVector[1]=0;
        thisVector[2]=0;
      }

      // update iterator value to this vector direction
      paImageIterator.SetInput(thisVector)

      // increment iterators
      ++paImageIterator;
      ++inputImageIterator;
    }
    

    // Done
    return 0;
}
