#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageToVTKImageFilter.h"
#include "itkFlipImageFilter.h"
#include "itkImageFileWriter.h"
#include "itkDiffusionTensor3D.h"
#include "itkSymmetricEigenAnalysisImageFilter.h"
#include "itkTensorFractionalAnisotropyImageFilter.h"
#include "itkVector.h"
#include "itkImageIterator.h"

#include "vtkSmartPointer.h"
#include "vtkImageSliceMapper.h"
#include "vtkImageActor.h"
#include "vtkRenderer.h"
#include "vtkRenderWindow.h"
#include "vtkRenderWindowInteractor.h"
#include "vtkCamera.h"
#include "vtkInteractorStyleImage.h"
#include "vtkCommand.h"
#include "vtkImageSlice.h"
#include "vtkImageProperty.h"
#include "vtkDiscreteMarchingCubes.h"
#include "vtkPolyDataMapper.h"
#include "vtkPoints.h"
#include "vtkNew.h"
#include "vtkProgrammableFilter.h"
#include "vtkCallbackCommand.h"
#include "vtkProperty.h"
#include "vtkSphereSource.h"
#include "vtkPolyPointSource.h"
#include "vtkTextActor.h"
#include "vtkTextProperty.h"

//Global Types
const unsigned int nDims = 3 ;   // Setup types
typedef itk::DiffusionTensor3D < double > DiffTensorType ;
typedef itk::Image < DiffTensorType , nDims > ImageType ;
typedef itk::Vector <double, nDims> VectorType;
typedef itk::Image <VectorType, nDims> PAImageType ;
typedef itk::Image < double, nDims> FAImageType ;
typedef itk::Image < int, nDims> TrackImageType ;
typedef itk::Image < int, nDims> SegmentedImageType ;

//Global Variables
vtkNew<vtkPoints> points;
std::list<ImageType::IndexType> globalList;

namespace {
void TimerCallbackFunction(vtkObject* caller, long unsigned int eventId,
                           void* clientData, void* callData);

unsigned int counter = 0;
unsigned int maxCount = 1000000;

void AdjustPoints(void* arguments)
{
   vtkProgrammableFilter* programmableFilter = static_cast<vtkProgrammableFilter*>(arguments);
  
   vtkPoints* inPts = programmableFilter->GetPolyDataInput()->GetPoints();
   vtkIdType numPts = inPts->GetNumberOfPoints();
   vtkNew<vtkPoints> newPts;
   newPts->SetNumberOfPoints(numPts);

   double p[3];
   for (int i=0; i<numPts; i++){
     if (i < counter+1){
     points->GetPoint(i, p);
     newPts->SetPoint(i, p);
     }
     else {
      inPts->GetPoint(i,p);
      newPts->SetPoint(i,p);
     }
   }

    programmableFilter->GetPolyDataOutput()->CopyStructure(programmableFilter->GetPolyDataInput());
    programmableFilter->GetPolyDataOutput()->SetPoints(newPts);
    std::cout<<"Iteration: " << counter << ", Number of Points: "<< numPts << std::endl;
}

} // namespace

namespace {
void TimerCallbackFunction(vtkObject* caller,
                           long unsigned int vtkNotUsed(eventId),
                           void* clientData, void* vtkNotUsed(callData))
{

  auto programmableFilter = static_cast<vtkProgrammableFilter*>(clientData);

  vtkRenderWindowInteractor* iren =
      static_cast<vtkRenderWindowInteractor*>(caller);

  programmableFilter->Modified();

  iren->Render();

  if (counter > maxCount)
  {
    iren->DestroyTimer();
  }

  counter++;
}
} // namespace


struct myLists
{
  std::list<ImageType::IndexType> seedtrackPixelList;
  std::list<ImageType::IndexType> segmenttrackPixelList;
} ;

int RecursiveTrack(PAImageType::Pointer myPAImage, FAImageType::Pointer myFAImage, TrackImageType::Pointer myTrackImage, ImageType::IndexType thisloc, double delta, int iter, std::list<ImageType::IndexType> & trackPixelList)
{
  // Check Stopping Conditions
  if (!myTrackImage->GetLargestPossibleRegion().IsInside(thisloc)) { //Voxel outside of image
    return 0;
  }
  if (myTrackImage->GetPixel(thisloc) == 1.0) { //Voxel already visited
    return 0 ;
  }
  if (myFAImage->GetPixel(thisloc) < 0.25) { //Voxel below minFA
    return 0;
  }
  if (iter > 1000) { //Surpassed maximum iterations
    return 0;
  }

  // If Stopping conditions pass, set voxel intensity
  myTrackImage->SetPixel(thisloc , 1.0) ;
  //std::cout << thisloc << std::endl;
  trackPixelList.push_back(thisloc);
  iter++;

  // Get next forward and back pixel
  ImageType::IndexType newlocplus = thisloc ;
  ImageType::IndexType newlocneg = thisloc ;
  
  VectorType thisVector ;
  thisVector = myPAImage->GetPixel(thisloc);
  newlocplus [0] = round(thisVector[0]*delta + thisloc[0]);
  newlocplus [1] = round(thisVector[1]*delta + thisloc[1]);
  newlocplus [2] = round(thisVector[2]*delta + thisloc[2]);
  RecursiveTrack(myPAImage, myFAImage, myTrackImage, newlocplus, delta, iter, trackPixelList);
  
  newlocneg [0] = round(-thisVector[0]*delta + thisloc[0]);
  newlocneg [1] = round(-thisVector[1]*delta + thisloc[1]);
  newlocneg [2] = round(-thisVector[2]*delta + thisloc[2]);
  RecursiveTrack(myPAImage, myFAImage, myTrackImage, newlocneg , delta, iter, trackPixelList);

  return 0;
}

FAImageType::Pointer makeFAImage(char * DTIFileName) 
{
  typedef itk::ImageFileReader < ImageType > ImageReaderType1 ;
  ImageReaderType1 ::Pointer myReader1 = ImageReaderType1 ::New() ;  
  myReader1->SetFileName (DTIFileName) ;
  myReader1->Update();   // go read
  ImageType::Pointer myImage = myReader1->GetOutput();
  
  //---(3) Compute FA---//
  typedef itk::TensorFractionalAnisotropyImageFilter <ImageType, FAImageType> FAFilterType ;
  FAFilterType::Pointer myFAImageFilter = FAFilterType::New() ;
  myFAImageFilter ->SetInput(myImage) ;
  myFAImageFilter ->Update() ;

  return myFAImageFilter->GetOutput();
}

myLists makeTracks(char * DTIFileName,  char * SegmentCCFileName, ImageType::IndexType seed)
{
  typedef itk::ImageFileReader < ImageType > ImageReaderType1 ;
  ImageReaderType1 ::Pointer myReader1 = ImageReaderType1 ::New() ;  
  myReader1->SetFileName (DTIFileName) ;
  myReader1->Update();   // go read
  ImageType::Pointer myImage = myReader1->GetOutput();

  typedef itk::ImageFileReader < SegmentedImageType > ImageReaderType2 ;
  ImageReaderType2 ::Pointer myReader2 = ImageReaderType2::New() ;  
  myReader2 ->SetFileName ( SegmentCCFileName ) ;
  myReader2 ->Update();   // go read
  SegmentedImageType::Pointer SegmentedCCImage = myReader2->GetOutput();
   
  //---(2) Compute principal eigenvector---//
  ImageType::RegionType myCustomRegion;
  ImageType::SizeType size = myImage->GetLargestPossibleRegion().GetSize() ;
  ImageType::IndexType corner ;
  corner[0] = 0; corner[1] = 0; corner[2] = 0;
  //std::cout << "Size: "<< size << std::endl;
  myCustomRegion.SetSize( size ) ;
  myCustomRegion.SetIndex( corner ) ;
  typedef itk::ImageRegionIterator < ImageType > InputIteratorType ;
  typedef itk::ImageRegionIterator < PAImageType > OutputIteratorType ;

  PAImageType::Pointer myPAImage = PAImageType::New() ;
  myPAImage->SetOrigin(myImage->GetOrigin() ) ;
  myPAImage->SetDirection(myImage->GetDirection() );
  myPAImage->SetSpacing(myImage->GetSpacing() );
  myPAImage->SetRegions(myCustomRegion);
  myPAImage->Allocate() ;

  OutputIteratorType outputIterator (myPAImage, myCustomRegion);
  InputIteratorType inputIterator (myImage, myCustomRegion);
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

  //---(3) Compute FA---//
  typedef itk::TensorFractionalAnisotropyImageFilter <ImageType, FAImageType> FAFilterType ;
  FAFilterType::Pointer myFAImageFilter = FAFilterType::New() ;
  myFAImageFilter ->SetInput(myImage) ;
  myFAImageFilter ->Update() ;
  
    //---(4) Seed Voxel Track---//
  TrackImageType::Pointer myTrackImage = TrackImageType::New() ;
  myTrackImage->SetOrigin(myImage->GetOrigin() ) ;
  myTrackImage->SetDirection(myImage->GetDirection() );
  myTrackImage->SetSpacing(myImage->GetSpacing() );
  myTrackImage->SetRegions(myCustomRegion);
  myTrackImage->Allocate() ;

  int iter = 0;
  double delta = 0.85;
  ImageType::IndexType thisloc = seed;
  ImageType::IndexType newloc = seed;
  
  std::list<ImageType::IndexType> seedtrackPixelList;
  RecursiveTrack(myPAImage, myFAImageFilter->GetOutput(), myTrackImage, thisloc, delta, iter, seedtrackPixelList);
  
  //---(10) Segmented corpus callosum voxel track---//
  TrackImageType::Pointer myCCTrackImage = TrackImageType::New() ;
  myCCTrackImage ->SetOrigin(myImage->GetOrigin() ) ;
  myCCTrackImage ->SetDirection(myImage->GetDirection() );
  myCCTrackImage ->SetSpacing(myImage->GetSpacing() );
  myCCTrackImage ->SetRegions(myCustomRegion);
  myCCTrackImage ->Allocate() ;

  typedef itk::ImageRegionIterator < SegmentedImageType > SegmentIteratorType ;
  SegmentIteratorType segmentIterator (SegmentedCCImage, myCustomRegion);
  segmentIterator .GoToBegin ();
  
  std::list<ImageType::IndexType> segmenttrackPixelList;

  while (!segmentIterator.IsAtEnd() )
  {
   if (segmentIterator.Value() == 1.0)
   {
      int iter = 0;
      RecursiveTrack(myPAImage, myFAImageFilter->GetOutput(), myCCTrackImage , segmentIterator.GetIndex(), delta, iter, segmenttrackPixelList);
   }
   
   ++segmentIterator ;
  }

  myLists result = {seedtrackPixelList, segmenttrackPixelList};
  return result;
}


int main ( int argc, char * argv[] )
{
  // Verify command line arguments
  if( argc < 4 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputDTIImageFile inputMask inputTrackOption" << std::endl ; 
      std::cerr <<"Input 1 for seed track, Input 2 for segmentation track" << std::endl;
      return -1 ;
    }
  int seedOrSeg = strtol(argv[3], NULL, 10);

  //---PART1---//
  // Runs Part 1 of the Assignment (without writing new image files) with 3 functions
  // Function input: arg[1]                             output: FA ITK Image
  // Function input: arg[1], arg[2], seed location,     output: list of seedtrackPixels and segmenttrackPixels
  
  FAImageType::Pointer myITKImage = makeFAImage(argv[1]);
  ImageType::IndexType seed ;
  seed[0] = 73; seed[1] = 89; seed[2] = 38;
  myLists myResults = makeTracks(argv[1], argv[2], seed); 

  ImageType::RegionType myCustomRegion;
  ImageType::SizeType size = myITKImage->GetLargestPossibleRegion().GetSize() ;
  ImageType::IndexType corner ;
  corner[0] = 0; corner[1] = 0; corner[2] = 0;
  myCustomRegion.SetSize( size ) ;
  myCustomRegion.SetIndex( corner ) ;
  
  //---PART2---//
  //---ITK---// 
  TrackImageType::Pointer myTrackImage = TrackImageType::New() ;
  myTrackImage->SetOrigin(myITKImage->GetOrigin() ) ;
  myTrackImage->SetDirection(myITKImage->GetDirection() );
  myTrackImage->SetSpacing(myITKImage->GetSpacing() );
  myTrackImage->SetRegions(myCustomRegion);
  myTrackImage->Allocate() ;
  myTrackImage->SetPixel(seed , 1.0) ;

  
  // Flip Image
  typedef itk::FlipImageFilter < FAImageType > FIFType ;
  FIFType::Pointer myFlip = FIFType::New();
  myFlip->SetInput(myITKImage);
  FIFType::FlipAxesArrayType flipAxes;
  flipAxes[0] = true; flipAxes[1] = true; flipAxes[2] = false;
  myFlip->SetFlipAxes(flipAxes);

  // Connect ITK portion to the VTK portion
  typedef itk::ImageToVTKImageFilter < FAImageType > ITKToVTKFilterType ;
  ITKToVTKFilterType::Pointer itkToVTKfilter = ITKToVTKFilterType::New() ;
  itkToVTKfilter->SetInput ( myFlip->GetOutput() ) ;
  itkToVTKfilter->Update() ;
  
  typedef itk::ImageToVTKImageFilter < SegmentedImageType > ITKMaskToVTKFilterType ;
  ITKMaskToVTKFilterType::Pointer itkMaskToVTKfilter = ITKMaskToVTKFilterType::New() ;
  itkMaskToVTKfilter->SetInput ( myTrackImage) ;
  itkMaskToVTKfilter->Update() ;
  

  //---VTK---//
  
  //---IMAGESLICE---//
  vtkSmartPointer < vtkImageSliceMapper > imageMapper = vtkSmartPointer < vtkImageSliceMapper > ::New() ;
  imageMapper->SetInputData ( itkToVTKfilter->GetOutput() ) ;
  imageMapper->SetOrientationToX () ;
  imageMapper->SetSliceNumber ( 55 ) ;
  imageMapper->SliceAtFocalPointOn () ;
  imageMapper->SliceFacesCameraOn () ;  
  vtkSmartPointer <vtkImageSlice> slice = vtkSmartPointer <vtkImageSlice> ::New();
  slice->SetMapper(imageMapper);
  slice->GetProperty()->SetColorWindow(0.8879);
  slice->GetProperty()->SetColorLevel(0.4440);
  
  //---POLYPOINTSOURCE---//
  if (seedOrSeg == 1){
      globalList = myResults.seedtrackPixelList;
  }
  if (seedOrSeg == 2){
      globalList = myResults.segmenttrackPixelList;
  }
  vtkNew<vtkPolyPointSource> polypointSource;

  ImageType::IndexType thisIndex;
  for (std::list<ImageType::IndexType>::iterator it = globalList.begin(); it!=globalList.end(); it++){
    thisIndex = *it;
    points->InsertNextPoint(thisIndex[0], thisIndex[1], thisIndex[2]);
  }
  
  vtkNew<vtkPoints> initialPoints;
  std::list<ImageType::IndexType>::iterator it = globalList.begin();
  thisIndex = *it;
  for (std::list<ImageType::IndexType>::iterator it = globalList.begin(); it!=globalList.end(); it++){
    initialPoints->InsertNextPoint(thisIndex[0], thisIndex[1], thisIndex[2]);
  }


  polypointSource->SetPoints(initialPoints);
  polypointSource->Update();
  vtkNew<vtkProgrammableFilter> programmableFilter;
  programmableFilter->SetInputConnection(polypointSource->GetOutputPort());
  programmableFilter->SetExecuteMethod(AdjustPoints, programmableFilter);
  
  vtkSmartPointer < vtkPolyDataMapper > polyMapper = vtkSmartPointer < vtkPolyDataMapper > ::New() ;
  polyMapper->SetInputConnection (programmableFilter->GetOutputPort()) ;
  vtkSmartPointer < vtkActor > actor = vtkSmartPointer < vtkActor >::New() ;
  actor->SetMapper(polyMapper);

  vtkSmartPointer < vtkTextActor > text1 = vtkSmartPointer < vtkTextActor >::New() ;
  text1->SetInput ( "FA Image" ) ;
  text1->SetDisplayPosition ( 50, 450 ) ;
  text1->GetTextProperty()->SetColor ( 1, 1, 1 ) ;
  text1->GetTextProperty()->SetFontSize ( 20 ) ;

  vtkSmartPointer < vtkTextActor > text2 = vtkSmartPointer < vtkTextActor >::New() ;
    if (seedOrSeg == 1){
      text2->SetInput ( "Seed Track (click to view)" ) ;
  }
  if (seedOrSeg == 2){
      text2->SetInput ( "Segment Track (click to view)" ) ;
  }
  text2->SetDisplayPosition ( 550, 450 ) ;
  text2->GetTextProperty()->SetColor ( 1, 1, 1 ) ;
  text2->GetTextProperty()->SetFontSize ( 20 ) ;

  //---RENDER1---//
  vtkSmartPointer < vtkRenderer > renderer1 = vtkSmartPointer < vtkRenderer >::New() ;
  renderer1->AddViewProp ( slice ) ;
  renderer1->SetViewport(0,0,0.5,1);
  renderer1->AddActor(text1);
    
	  //---RENDER1CAMERA---//
	  vtkSmartPointer < vtkCamera > camera = renderer1->GetActiveCamera() ;

	  double position[3],  imageCenter[3] ;
	  itkToVTKfilter->GetOutput()->GetCenter ( imageCenter ) ;
	  position[0] = imageCenter[0] ;
	  position[1] = imageCenter[1] ;
	  position[2] = -160 ;
	  double spacing[3] ;
	  int imageDims[3] ;
	  itkToVTKfilter->GetOutput()->GetSpacing ( spacing ) ;
	  itkToVTKfilter->GetOutput()->GetDimensions ( imageDims ) ;
	  double imagePhysicalSize[3] ;
	  for ( unsigned int d = 0 ; d < 3 ; d++ )
	    {
	      imagePhysicalSize[d] = spacing[d] * imageDims[d] ;
	    }
	  camera->ParallelProjectionOn () ; 
	  camera->SetFocalPoint ( imageCenter ) ;
	  camera->SetPosition ( position ) ;
	  camera->SetParallelScale ( imageDims[2] / 0.8 ) ;

  //---RENDER2---//
  vtkSmartPointer < vtkRenderer > renderer2 = vtkSmartPointer < vtkRenderer >::New() ;
  renderer2->SetViewport(0.5, 0, 1, 1);
  renderer2->AddActor(actor);
  renderer2->AddActor(text2);
  
  	//---RENDER2CAMERA---//
    double thisIndexFP[3], thisIndexPosition[3];
    thisIndexFP[0] = thisIndex[0]; thisIndexFP[1] = thisIndex[1]; thisIndexFP[2] = thisIndex[2];
    thisIndexPosition[0] = thisIndexFP[0]; thisIndexPosition[1] = thisIndexFP[1];
    thisIndexPosition[2] = -160;
  	renderer2->ResetCamera();
  	camera = renderer2->GetActiveCamera();
    camera->ParallelProjectionOn () ; 
    camera->SetFocalPoint ( thisIndexFP ) ;
    camera->SetPosition ( thisIndexPosition ) ;
    camera->SetParallelScale ( imageDims[2] / 0.8 ) ;
  
  //---WINDOW---//
  vtkSmartPointer < vtkRenderWindow > window = vtkSmartPointer < vtkRenderWindow >::New() ;
  window->AddRenderer ( renderer1 ) ;
  window->AddRenderer ( renderer2 );
  window->SetSize ( 1000, 500 ) ;
  window->Render();

  vtkSmartPointer < vtkRenderWindowInteractor > interactor = vtkSmartPointer < vtkRenderWindowInteractor >::New() ;
  interactor->SetRenderWindow ( window ) ;

  vtkSmartPointer < vtkInteractorStyleImage > style = vtkSmartPointer < vtkInteractorStyleImage >::New() ;
  style->SetInteractionModeToImageSlicing() ;

  interactor->Initialize() ;
  interactor->CreateRepeatingTimer ( 0.5 ) ;

  vtkNew<vtkCallbackCommand> timerCallback;
  timerCallback->SetCallback(TimerCallbackFunction);
  timerCallback->SetClientData(programmableFilter);

  interactor->AddObserver(vtkCommand::TimerEvent, timerCallback);

  // Run
  interactor->Start() ;
  interactor->ReInitialize();

  // Done.
  return 0 ;
}