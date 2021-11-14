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
typedef itk::ImageFileReader < ImageType > TensorImageFileReaderType ;
typedef itk::ImageRegionIterator < ImageType > InputImageIterator ;
typedef itk::ImageRegionIterator < PAImageType > PAImageIterator ;
typedef itk::TensorFractionalAnisotropyImageFilter <ImageType, BaseImageType> FAImageFilterType;


ImageType::IndexType computeNewIdx(VectorType thisVector, double delta, ImageType::IndexType currLoc, bool sign){
  ImageType::IndexType newLoc = currLoc;

  if (sign){
    newLoc[0] = round(thisVector[0]*delta + currLoc[0]);
    newLoc[1] = round(thisVector[1]*delta + currLoc[1]);
    newLoc[2] = round(thisVector[2]*delta + currLoc[2]);

    return newLoc;
  }

  // return for negative
  newLoc[0] = round(-thisVector[0]*delta + currLoc[0]);
  newLoc[1] = round(-thisVector[1]*delta + currLoc[1]);
  newLoc[2] = round(-thisVector[2]*delta + currLoc[2]);
  
  return newLoc;
}

int traverseImage(BaseImageType::Pointer faImage, PAImageType::Pointer paImage, BaseImageType::Pointer trackerImage, ImageType::IndexType curLoc ,double delta, int iter){
  // stopping conditions
  // if location is outside of the image
  if (!trackerImage -> GetLargestPossibleRegion().IsInside(curLoc)){
    return 0;
  }

  // if iter is greater than 1000
  if (iter > 1000){
    return 0;
  }

  // if location is already visited
  if (trackerImage -> GetPixel(curLoc) == 1){
    return 0;
  }

  // if the eigen value is less then FA val
  if (faImage -> GetPixel(curLoc) < 0.2) {
    return 0;
  }

  // mark the pixel in tracker image as visited
  trackerImage -> SetPixel(curLoc, 1.0);
  iter++;

  VectorType thisVector ;
  thisVector = paImage->GetPixel(curLoc);

  ImageType::IndexType forward = computeNewIdx(thisVector, delta, curLoc, true);
  ImageType::IndexType backward = computeNewIdx(thisVector, delta, curLoc, true);

  // make recursive calls to track image
  traverseImage(faImage, paImage, trackerImage, forward, delta, iter);
  traverseImage(faImage, paImage, trackerImage, backward, delta, iter);

}


int main ( int argc, char * argv[] )
{
  // --- Verify command line arguments----//
  if( argc < 6 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputFileImage inputSegmentFile outputEigenVecFile outputFAImage outputSegmentTrackFile" << std::endl ; 
      return -1 ;
    }

  
  // read input image file
  TensorImageFileReaderType ::Pointer inputFileReader = TensorImageFileReaderType ::New() ;  
  inputFileReader->SetFileName ( argv[1] ) ;
  inputFileReader->Update();

  // define the pointer for image reader
  ImageType::Pointer img = inputFileReader->GetOutput();

  // compute new region which would help in computation of the eigen values and vector
  ImageType::RegionType newRegion ; // define new region for iterator

  ImageType::SizeType size = img->GetLargestPossibleRegion().GetSize() ;
  ImageType::IndexType origin ; // create a new index pointing to origin
  origin[0] = 0; origin[1] = 0; origin[2] = 0;

  newRegion.SetSize( size ) ;
  newRegion.SetIndex( origin ) ;
  

  // create a tracker image for the segmentation of the index it would be used later
  BaseImageType::Pointer trackerImage = BaseImageType::New() ;
  trackerImage->SetOrigin(img->GetOrigin() ) ;
  trackerImage->SetDirection(img->GetDirection() );
  trackerImage->SetSpacing(img->GetSpacing() );
  trackerImage->SetRegions(newRegion);
  trackerImage->Allocate() ;


  // define PA image for other ops such as eigen value computation
  PAImageType::Pointer paImage = PAImageType::New() ;
  paImage->SetOrigin(img->GetOrigin() ) ;
  paImage->SetDirection(img->GetDirection() );
  paImage->SetSpacing(img->GetSpacing() );
  paImage->SetRegions(newRegion);
  paImage->Allocate() ;


  // define tensors and vectors to store eigen val and vector
  TensorType thisTensor ;
  TensorType::EigenValuesArrayType eigenValArrayType;
  TensorType::EigenVectorsMatrixType eigenValMatrixType ;
  VectorType thisVector ;

  PAImageIterator paIterator (paImage, newRegion);
  InputImageIterator inputImageIterator (img, newRegion);
  paIterator.GoToBegin ();
  inputImageIterator.GoToBegin () ;

  while (!paIterator.IsAtEnd() )
  {
   thisTensor =  inputImageIterator.Value() ;
   thisTensor.ComputeEigenAnalysis(eigenValArrayType, eigenValMatrixType) ;

   // compute and store eigenValue in this tensor to update it to next iterator
   thisVector[0] = eigenValMatrixType[2][0]*1 ; 
   thisVector[1] = eigenValMatrixType[2][1]*1 ; 
   thisVector[2] = eigenValMatrixType[2][2]*1 ;


  // change the vector to zero for making it to zero for the out of bound eigen values
   if (eigenValMatrixType[2][2] == 1) { 
     thisVector[0] = 0; 
     thisVector[1] = 0; 
     thisVector[2] = 0;
   }

   // update the iterators for making it better 
   paIterator.Set(thisVector) ;
   ++paIterator ;
   ++inputImageIterator ;
  }

  // compute FA for the image
  FAImageFilterType::Pointer faImageFilter = FAImageFilterType::New();
  // pass input image as input
  faImageFilter -> SetInput(img);
  faImageFilter -> Update();


  //------Voxel Tracking ------//

  // define random seed
  ImageType::IndexType seed;

  seed[0] = 69;
  seed[1] = 75;
  seed[2] = 31;

  // define hyperparms for tracking the inputs
  unsigned int iter=0;
  double delta = 0.9;

  ImageType::IndexType currLoc = seed;
  ImageType::IndexType newLoc = seed;

  traverseImage(faImageFilter -> GetOutput(), paImage, trackerImage, currLoc, delta, iter);

  // work on segmented seed voxel
  typedef itk::ImageFileWriter < PAImageType> ImageWriterType1 ;   
  ImageWriterType1 ::Pointer myWriter1 = ImageWriterType1::New();   
  myWriter1->SetFileName( argv[3] );   
  myWriter1->SetInput(paImage);  
  myWriter1->Update();   

  typedef itk::ImageFileWriter < BaseImageType> ImageWriterType2 ;   
  ImageWriterType2 ::Pointer myWriter2 = ImageWriterType2::New();   
  myWriter2->SetFileName( argv[4] );   
  myWriter2->SetInput(faImageFilter->GetOutput() );  
  myWriter2->Update();   

  // typedef itk::ImageFileWriter < BaseImageType> ImageWriterType3 ;   
  // ImageWriterType3 ::Pointer myWriter3 = ImageWriterType3::New();   
  // myWriter3->SetFileName( argv[5] );   
  // myWriter3->SetInput(trackerImage );  
  // myWriter3->Update();

  // Done.
  return 0 ;
}
