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
typedef itk::ImageFileReader < BaseImageType > BaseImageReaderType;
typedef itk::ImageRegionIterator < ImageType > InputImageIterator ;
typedef itk::ImageRegionIterator < PAImageType > PAImageIterator ;
typedef itk::ImageRegionIterator <BaseImageType> BaseImageIteratorType ;
typedef itk::TensorFractionalAnisotropyImageFilter <ImageType, BaseImageType> FAImageFilterType;

// function to create a tracker image
int CreateTrackerImage(BaseImageType::Pointer tracker, ImageType::Pointer img, ImageType::RegionType region){
  tracker->SetOrigin(img->GetOrigin() ) ;
  tracker->SetDirection(img->GetDirection() );
  tracker->SetSpacing(img->GetSpacing() );
  tracker->SetRegions(region);
  tracker->Allocate() ;

  return 0;
}

// Function to return the new index for the tracker
ImageType::IndexType computeNewIdx(VectorType thisVector, double delta, ImageType::IndexType currLoc, bool sign){
  ImageType::IndexType newLoc = currLoc;

  // check if forward side of index required or something different
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

  // find the new set of forward and backward indexes to call the function
  ImageType::IndexType forward = computeNewIdx(thisVector, delta, curLoc, true);
  ImageType::IndexType backward = computeNewIdx(thisVector, delta, curLoc, true);

  // make recursive calls to track image
  traverseImage(faImage, paImage, trackerImage, forward, delta, iter);
  traverseImage(faImage, paImage, trackerImage, backward, delta, iter);

}

// a generic file writer class which takes image pointer and filename as input
// and writes the file to it.
template <class ImageType>
int imageWriter(typename ImageType::Pointer image, char* filename) {

  // create an ITK file writer
  typedef itk::ImageFileWriter <ImageType> ImageFileWriterType;
  typename ImageFileWriterType::Pointer myFileWriter = ImageFileWriterType::New();

  // perform ops for writing the image
  myFileWriter -> SetFileName(filename);
  myFileWriter -> SetInput(image);
  myFileWriter -> Update();

  return 0;

}

int main ( int argc, char * argv[] )
{
  // --- Verify command line arguments----//
  if( argc < 6 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputFileImage inputSegmentedFile outputEigenVecFile outputFAImage outputTrackerFile" << std::endl ; 
      return -1 ;
    }

  
  // read input image file
  std::cout << "Loading Input TensorImage" <<std::endl;
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
  // trackerImage->SetOrigin(img->GetOrigin() ) ;
  // trackerImage->SetDirection(img->GetDirection() );
  // trackerImage->SetSpacing(img->GetSpacing() );
  // trackerImage->SetRegions(newRegion);
  // trackerImage->Allocate() ;
  std::cout << "Creating a Tracker Image for the Input Image" <<std::endl;
  CreateTrackerImage(trackerImage, img, newRegion);

  // ---- EigenValue and FA computation ------//

  // define PA image for other ops such as eigen value computation
  std::cout << "Allocating Principal Axis Eigen Value Image" <<std::endl;
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

  std::cout << "Computing FA for the Input Image" <<std::endl;

  // compute FA for the image
  FAImageFilterType::Pointer faImageFilter = FAImageFilterType::New();
  // pass input image as input
  faImageFilter -> SetInput(img);
  faImageFilter -> Update();


  //------Voxel Tracking ------//
  std::cout << "Performing Voxel tracking for the Input Image" <<std::endl;
  // define random seed
  ImageType::IndexType seed;

  seed[0] = 69;
  seed[1] = 89;
  seed[2] = 36;

  // define hyperparms for tracking the inputs
  unsigned int iter=0;
  double delta = 0.9;


  // define location of the seed
  ImageType::IndexType currLoc = seed;
  ImageType::IndexType newLoc = seed;

  // start image traversal for the images
  traverseImage(faImageFilter -> GetOutput(), paImage, trackerImage, currLoc, delta, iter);
  
  // ---- Perform Segmentation Task ---- //
  std::cout << "Loading Segmentation Image" <<std::endl;
  // load segmented image file
  BaseImageReaderType::Pointer segmentedFileReader = BaseImageReaderType::New();
  segmentedFileReader -> SetFileName(argv[2]);
  segmentedFileReader -> Update(); // read the file
  BaseImageType::Pointer segmentedImage = segmentedFileReader -> GetOutput();

  // create a tracker image for the segmented file
  std::cout << "Allocating Tracker image for Segmentation Image" <<std::endl;
  BaseImageType::Pointer segmentedTrackerImage = BaseImageType::New();
  CreateTrackerImage(segmentedTrackerImage, img, newRegion);

  // create segmentation iterator
  std::cout << "Computing Tracker Image for the Segmented Image" <<std::endl;
  BaseImageIteratorType segmentationIter (segmentedImage, newRegion);
  segmentationIter.GoToBegin();

  while (!segmentationIter.IsAtEnd())
  {
    if (segmentationIter.Value() == 1.0){
      int iter = 0;
      std::cout << "Segmented Image iter " << iter <<std::endl;
      traverseImage(faImageFilter -> GetOutput(), paImage, segmentedTrackerImage, segmentationIter.GetIndex(), delta, iter);
    }
  }
  

  std::cout << "Writing Output Images" <<std::endl;
  imageWriter<PAImageType>(paImage, argv[3]);
  imageWriter<BaseImageType>(faImageFilter->GetOutput(), argv[4]);
  imageWriter<BaseImageType>(trackerImage, argv[5]);

  // Done.
  return 0 ;
}
