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
#include "vtkImageSlice.h"
#include "vtkNew.h"
#include "vtkInteractorStyleImage.h"
#include "vtkProgrammableFilter.h"
#include "vtkCallbackCommand.h"

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
typedef itk::ImageToVTKImageFilter <BaseImageType> BaseImageToVTKFilterType;
typedef itk::ImageToVTKImageFilter <PAImageType> PAImageToVTKFilterType;
typedef itk::ImageToVTKImageFilter <ImageType> ImageToVTKFilterType;

// --- Global Variables -- //
vtkNew<vtkPoints> points;
std::list<ImageType::IndexType> globalList;

namespace {
void MyTimerCallBack(vtkObject* caller, long unsigned int eventId,
                           void* clientData, void* callData);

unsigned int count = 0;
unsigned int maxCount = 1000000;

void AdjustPoints(void* arguments)
{
   vtkProgrammableFilter* myFilter = static_cast<vtkProgrammableFilter*>(arguments);
  
   vtkPoints* inPts = myFilter->GetPolyDataInput()->GetPoints();
   vtkIdType numPts = inPts->GetNumberOfPoints();
   vtkNew<vtkPoints> nPts;
   nPts->SetNumberOfPoints(numPts);

   double pt[3];
   for (int i=0; i<numPts; i++){
     if (i < count+1){
     points->GetPoint(i, pt);
     nPts->SetPoint(i, pt);
     }
     else {
      inPts->GetPoint(i,pt);
      nPts->SetPoint(i,pt);
     }
   }

    myFilter->GetPolyDataOutput()->CopyStructure(myFilter->GetPolyDataInput());
    myFilter->GetPolyDataOutput()->SetPoints(nPts);
}

} // namespace

namespace {
void MyTimerCallBack(vtkObject* caller,
                           long unsigned int vtkNotUsed(eventId),
                           void* clientData, void* vtkNotUsed(callData))
{

  auto programmableFilter = static_cast<vtkProgrammableFilter*>(clientData);

  vtkRenderWindowInteractor* iren =
      static_cast<vtkRenderWindowInteractor*>(caller);

  programmableFilter->Modified();

  iren->Render();

  if (count > maxCount)
  {
    iren->DestroyTimer();
  }

  count++;
}
} // namespace

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

int traverseImage(BaseImageType::Pointer faImage, PAImageType::Pointer paImage, BaseImageType::Pointer trackerImage, ImageType::IndexType curLoc ,double delta, int iter, std::list<ImageType::IndexType> & trackerList){
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
  trackerList.push_back(curLoc);
  iter++;

  VectorType thisVector ;
  thisVector = paImage->GetPixel(curLoc);

  // find the new set of forward and backward indexes to call the function
  ImageType::IndexType forward = computeNewIdx(thisVector, delta, curLoc, true);
  ImageType::IndexType backward = computeNewIdx(thisVector, delta, curLoc, false);

  // make recursive calls to track image
  traverseImage(faImage, paImage, trackerImage, forward, delta, iter,trackerList);
  traverseImage(faImage, paImage, trackerImage, backward, delta, iter,trackerList);

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

  std::list<ImageType::IndexType> seedTrackerList;
  // start image traversal for the images
  traverseImage(faImageFilter -> GetOutput(), paImage, trackerImage, currLoc, delta, iter, seedTrackerList);
  
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

  std::list<ImageType::IndexType> segmentedTrackerList;

  while (!segmentationIter.IsAtEnd())
  {
    if (segmentationIter.Value() == 1.0){
      int iter = 0;
      traverseImage(faImageFilter -> GetOutput(), paImage, segmentedTrackerImage, segmentationIter.GetIndex(), delta, iter, segmentedTrackerList);
    }
    ++segmentationIter;
  }


  // reterive all the points and set it into a global list
  ImageType::IndexType thisIndex;
  for (std::list<ImageType::IndexType>::iterator it = segmentedTrackerList.begin(); it != segmentedTrackerList.end(); it++){
    thisIndex = *it;
    points -> InsertNextPoint(thisIndex[0], thisIndex[1], thisIndex[2]);
  }

  // create a current initializer
  vtkNew<vtkPoints> initialPoints;
  std::list<ImageType::IndexType>::iterator it = segmentedTrackerList.begin();
  thisIndex = *it;
  for (std::list<ImageType::IndexType>::iterator it = globalList.begin(); it!=globalList.end(); it++){
    initialPoints->InsertNextPoint(thisIndex[0], thisIndex[1], thisIndex[2]);
  }

  // create a poly data type for creation of mesh
  vtkNew<vtkPolyData> polyPoints;
  polyPoints -> SetPoints(initialPoints);
  // polyPoints -> Update();
  
  // vtkNew<vtkProgrammableFilter> programmableFilter;
  // programmableFilter->SetInputConnetion(polypointSource->GetOutputPort());
  // programmableFilter->SetExecuteMethod(AdjustPoints, programmableFilter);

  vtkNew<vtkProgrammableFilter> timerprogrammableFilter;
  timerprogrammableFilter->SetInputData(polyPoints);
  timerprogrammableFilter->SetExecuteMethod(AdjustPoints, timerprogrammableFilter);

  vtkSmartPointer < vtkPolyDataMapper > polyMapper = vtkSmartPointer < vtkPolyDataMapper > ::New() ;
  polyMapper->SetInputData (timerprogrammableFilter->GetPolyDataOutput()) ;

  vtkSmartPointer < vtkActor > actor = vtkSmartPointer < vtkActor >::New() ;
  actor->SetMapper(polyMapper);

  // create a scene that keeps track of all the actors in the workspace
  vtkSmartPointer < vtkRenderer > renderer = vtkSmartPointer < vtkRenderer >::New() ;
  renderer->AddActor ( actor ) ;
  renderer -> SetViewport(0.5, 0,1,1);

  BaseImageToVTKFilterType::Pointer faitkToVTKfilter = BaseImageToVTKFilterType::New() ;
  faitkToVTKfilter->SetInput ( faImageFilter -> GetOutput() ) ;
  faitkToVTKfilter->Update() ;
  faitkToVTKfilter->GetOutput() ;

  
  //---IMAGESLICE---//
  vtkSmartPointer < vtkImageSliceMapper > imageMapper = vtkSmartPointer < vtkImageSliceMapper > ::New() ;
  imageMapper->SetInputData ( faitkToVTKfilter->GetOutput() ) ;
  imageMapper->SetOrientationToX () ;
  imageMapper->SetSliceNumber ( 55 ) ;
  imageMapper->SliceAtFocalPointOn () ;
  imageMapper->SliceFacesCameraOn () ;  
  
  vtkSmartPointer <vtkImageSlice> slice = vtkSmartPointer <vtkImageSlice> ::New();
  slice->SetMapper(imageMapper);
  slice->GetProperty()->SetColorWindow(0.8879);
  slice->GetProperty()->SetColorLevel(0.4440);
  

  // VTK Portion of the code - visualization pipeline
  // mapper
  vtkSmartPointer < vtkImageSliceMapper > faimageMapper = vtkSmartPointer < vtkImageSliceMapper > ::New() ;
  faimageMapper->SetInputData ( faitkToVTKfilter->GetOutput() ) ;
  faimageMapper->SetOrientationToX () ;
  faimageMapper->SetSliceNumber ( 55 ) ;
  std::cout << "default for atfocalpoint: " << faimageMapper->GetSliceAtFocalPoint () << std::endl ;
  std::cout << "default for faces camera: " << faimageMapper->GetSliceFacesCamera () << std::endl ;
  faimageMapper->SliceAtFocalPointOn () ;
  faimageMapper->SliceFacesCameraOn () ;

  // slice
  vtkSmartPointer <vtkImageSlice> faImageSlicer = vtkSmartPointer <vtkImageSlice> ::New();
  faImageSlicer->SetMapper(faimageMapper);
  faImageSlicer->GetProperty()->SetColorWindow(0.8879);
  faImageSlicer->GetProperty()->SetColorLevel(0.4440);
  
  vtkSmartPointer < vtkRenderer > farenderer = vtkSmartPointer < vtkRenderer >::New() ;
  farenderer->AddViewProp ( slice ) ;
  farenderer->SetViewport(0,0,0.5,1);

  //---RENDER1CAMERA---//
	  vtkSmartPointer < vtkCamera > camera = farenderer->GetActiveCamera() ;

	  double position[3],  imageCenter[3] ;
	  faitkToVTKfilter->GetOutput()->GetCenter ( imageCenter ) ;
	  position[0] = imageCenter[0] ;
	  position[1] = imageCenter[1] ;
	  position[2] = -160 ;
	  double spacing[3] ;
	  int imageDims[3] ;
	  faitkToVTKfilter->GetOutput()->GetSpacing ( spacing ) ;
	  faitkToVTKfilter->GetOutput()->GetDimensions ( imageDims ) ;
	  double imagePhysicalSize[3] ;
	  for ( unsigned int d = 0 ; d < 3 ; d++ )
	    {
	      imagePhysicalSize[d] = spacing[d] * imageDims[d] ;
	    }
	  camera->ParallelProjectionOn () ; 
	  camera->SetFocalPoint ( imageCenter ) ;
	  camera->SetPosition ( position ) ;
	  camera->SetParallelScale ( imageDims[2] / 0.8 ) ;
  
  //---RENDER2CAMERA---//
    double thisIndexFP[3], thisIndexPosition[3];
    thisIndexFP[0] = thisIndex[0]; thisIndexFP[1] = thisIndex[1]; thisIndexFP[2] = thisIndex[2];
    thisIndexPosition[0] = thisIndexFP[0]; thisIndexPosition[1] = thisIndexFP[1];
    thisIndexPosition[2] = -160;
  	renderer->ResetCamera();
  	camera = renderer->GetActiveCamera();
    camera->ParallelProjectionOn () ; 
    camera->SetFocalPoint ( thisIndexFP ) ;
    camera->SetPosition ( thisIndexPosition ) ;
    camera->SetParallelScale ( imageDims[2] / 0.8 );
  
  //---WINDOW---//
  vtkSmartPointer < vtkRenderWindow > window = vtkSmartPointer < vtkRenderWindow >::New() ;
  window->AddRenderer ( farenderer ) ;
  window->AddRenderer ( renderer );
  window->SetSize ( 1000, 500 ) ;
  window->Render();

  vtkSmartPointer < vtkRenderWindowInteractor > interactor = vtkSmartPointer < vtkRenderWindowInteractor >::New() ;
  interactor->SetRenderWindow ( window ) ;

  vtkSmartPointer < vtkInteractorStyleImage > style = vtkSmartPointer < vtkInteractorStyleImage >::New() ;
  style->SetInteractionModeToImageSlicing() ;

  interactor->Initialize() ;
  interactor->CreateRepeatingTimer ( 0.5 ) ;

  vtkNew<vtkCallbackCommand> timerCallback;
  timerCallback->SetCallback(MyTimerCallBack);
  timerCallback->SetClientData(timerprogrammableFilter);

  interactor->AddObserver(vtkCommand::TimerEvent, timerCallback);

  // Run
  interactor->Start() ;
  interactor->ReInitialize();


  // Done.
  return 0 ;
}
