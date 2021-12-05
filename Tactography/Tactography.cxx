#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkDiffusionTensor3D.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkTensorFractionalAnisotropyImageFilter.h"
#include "itkVector.h"
#include "itkImageIterator.h"
#include "itkImageToVTKImageFilter.h"

#include "vtkSmartPointer.h"
#include "vtkImageSliceMapper.h"
#include "vtkImageActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkCamera.h"
#include "vtkPolyData.h"
#include "vtkPoints.h"
#include "vtkCellArray.h"
#include <vtkImageProperty.h>
#include "vtkInteractorStyleImage.h"
#include "vtkCommand.h"
#include "vtkCellData.h"
#include "vtkPolyDataMapper.h"
#include "vtkUnsignedCharArray.h"
#include "vtkRendererCollection.h"

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
typedef itk::ImageToVTKImageFilter <PAImageType> BaseImageToVTKFilterType;
typedef itk::ImageToVTKImageFilter <ImageType> ImageToVTKFilterType;


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
  if (trackerImage -> GetPixel(curLoc) == 1.0){
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
  ImageType::IndexType backward = computeNewIdx(thisVector, delta, curLoc, false);

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
  if( argc < 3 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputFileImage inputSegmentedFile" << std::endl ; 
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
  
  typedef itk::ImageToVTKImageFilter < faImageType > faITKToVTKFilterType ;
  faITKToVTKFilterType::Pointer faitkToVTKfilter = faITKToVTKFilterType::New() ;
  faitkToVTKfilter->SetInput ( myfaITKImage ) ;
  faitkToVTKfilter->Update() ;
  faitkToVTKfilter->GetOutput() ;
  
  // VTK Portion of the code - visualization pipeline
  // mapper
  vtkSmartPointer < vtkImageSliceMapper > imageMapper = vtkSmartPointer < vtkImageSliceMapper > ::New() ;
  imageMapper->SetInputData ( faitkToVTKfilter->GetOutput() ) ;
  imageMapper->SetOrientationToX () ;
  imageMapper->SetSliceNumber ( 55 ) ;
  std::cout << "default for atfocalpoint: " << imageMapper->GetSliceAtFocalPoint () << std::endl ;
  std::cout << "default for faces camera: " << imageMapper->GetSliceFacesCamera () << std::endl ;
  imageMapper->SliceAtFocalPointOn () ;
  imageMapper->SliceFacesCameraOn () ;

  vtkSmartPointer < vtkImageProperty > image_property = vtkSmartPointer <vtkImageProperty>::New() ;
  image_property->SetColorWindow(1.0) ;
  image_property->SetColorLevel(0.5) ;


  // Actor
  vtkSmartPointer < vtkImageActor > imageActor = vtkSmartPointer < vtkImageActor > ::New() ;
  imageActor->SetMapper ( imageMapper ) ;
  imageActor->SetProperty( image_property ) ; //pk
  imageActor->InterpolateOff(); //pk


  // Set up the scene, window, interactor
  vtkSmartPointer < vtkRenderer > renderer = vtkSmartPointer < vtkRenderer >::New() ;
  renderer->AddActor ( imageActor ) ;

  // get camera so we can position  
  vtkSmartPointer < vtkCamera > camera = renderer->GetActiveCamera() ;

  double position[3],  imageCenter[3] ;
  faItkToVtkFilter->GetOutput()->GetCenter ( imageCenter ) ;
  position[0] = imageCenter[0] ;
  position[1] = imageCenter[1] ;
  position[2] = -160 ;
  std::cout << "Image center: " << imageCenter[0] << " " << imageCenter[1] << " " << imageCenter[2] << std::endl ;
  double spacing[3] ;
  int imageDims[3] ;
  faItkToVtkFilter->GetOutput()->GetSpacing ( spacing ) ;
  faItkToVtkFilter->GetOutput()->GetDimensions ( imageDims ) ;
  double imagePhysicalSize[3] ;
  for ( unsigned int d = 0 ; d < 3 ; d++ )
    {
      imagePhysicalSize[d] = spacing[d] * imageDims[d] ;
    }


  camera->ParallelProjectionOn () ; 
  camera->SetFocalPoint ( imageCenter ) ;
  camera->SetPosition ( position ) ;
  //  camera->SetDistance ( imagePhysicalSize[2] * -1 ) ;
  std::cout << "Parallel scale: " << camera->GetParallelScale() << std::endl ;
  std::cout << imageDims[0] << " " << imageDims[1] << " " << imageDims[2] << std::endl ;
  camera->SetParallelScale ( imageDims[2] / 4 ) ;

  // set up window
  vtkSmartPointer < vtkRenderWindow > window = vtkSmartPointer < vtkRenderWindow >::New() ;
  window->AddRenderer ( renderer ) ;
  window->SetSize ( 500, 500 ) ;

  vtkSmartPointer < vtkRenderWindowInteractor > interactor = vtkSmartPointer < vtkRenderWindowInteractor >::New() ;
  interactor->SetRenderWindow ( window ) ;

  vtkSmartPointer < vtkInteractorStyleImage > style = vtkSmartPointer < vtkInteractorStyleImage >::New() ;
  //style->SetInteractionModeToImage3D() ;
  style->SetInteractionModeToImageSlicing() ;

  interactor->SetInteractorStyle ( style ) ;
  interactor->Initialize() ;

  // Polydata object for WM tract
  vtkSmartPointer<vtkPolyData> WMtract = vtkSmartPointer<vtkPolyData>::New() ;
  // Line, points and colors
  vtkSmartPointer<vtkPoints> points = vtkSmartPointer<vtkPoints>::New() ;
  WMtract->SetPoints( points ) ;
  vtkSmartPointer<vtkCellArray> lines = vtkSmartPointer<vtkCellArray>::New();
  WMtract->SetLines( lines ) ;
  vtkSmartPointer<vtkUnsignedCharArray> colors = vtkSmartPointer<vtkUnsignedCharArray>::New();
  colors->SetNumberOfComponents( 3 );
  WMtract->GetCellData()->SetScalars( colors ) ;
  // Mapper, actor and renderer
  vtkSmartPointer<vtkPolyDataMapper> wmmapper = vtkSmartPointer<vtkPolyDataMapper>::New();
  wmmapper->SetInputData(WMtract);
  vtkSmartPointer<vtkActor> wmactor = vtkSmartPointer<vtkActor>::New();
  wmactor->SetMapper(wmmapper);
  vtkSmartPointer < vtkRenderer > wmrenderer = vtkSmartPointer < vtkRenderer >::New() ;
  wmrenderer->AddActor(wmactor);
  wmrenderer->SetBackground(1, 1, 1);

  //  interactor->DestroyTimer ( timerId ) ;
  // run!
  interactor->Start() ;  

  //------Voxel Tracking ------//
  std::cout << "Performing Voxel tracking for the Input Image" <<std::endl;
  // define random seed
  ImageType::IndexType seed;

  seed[0] = 73;
  seed[1] = 83;
  seed[2] = 34;

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
      traverseImage(faImageFilter -> GetOutput(), paImage, segmentedTrackerImage, segmentationIter.GetIndex(), delta, iter);
    }
    ++segmentationIter;
  }
  
  // Done.
  return 0 ;
}
