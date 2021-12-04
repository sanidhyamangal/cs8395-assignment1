#include "itkImage.h"
#include "itkImageFileReader.h"
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

// Function to trace the tracts
/* int segmentTract(DImageType::IndexType cVoxel, wmtractImageType::Pointer wmtractImage, faImageType::Pointer faImage, pevImageType::Pointer pevImage, int *iteration_no)
{
   // Termination criteria
   if (*iteration_no == 100000)
   {  return 0;}
   else if (!wmtractImage->GetLargestPossibleRegion().IsInside(cVoxel))
   {  return 0;}
   else if ( faImage->GetPixel( cVoxel ) < 0.2)
   { return 0;}
   else if ( wmtractImage->GetPixel( cVoxel ) >= 1.0 )
   { return 0;}
   else
   {

   // At the voxel set the voxel to iteration number
   wmtractImage->SetPixel( cVoxel, *iteration_no );

   // Eigen vector at the voxel d[x]
   vectorType dx = pevImage->GetPixel( cVoxel );

   // Compute next voxel to visit
   DImageType::IndexType AB ;
   DImageType::IndexType BA ;
  for (int j = 0; j < 3; j++)
   {
    AB[j] = round(cVoxel[j] + dx[j] * 1)  ;
    BA[j] = round(cVoxel[j] - dx[j] * 1) ;
   }

   *iteration_no += 1;

   // Call function for backward and forward voxel
   segmentTract(BA, wmtractImage, faImage, pevImage, iteration_no);
   segmentTract(AB, wmtractImage, faImage, pevImage, iteration_no);

   return 0;
   }
}
 */
class CustomTimerCallback : public vtkCommand
{
public:
  static CustomTimerCallback * New ()
  {
    CustomTimerCallback *callback = new CustomTimerCallback ;
    return callback ;
  }

  virtual void Execute (vtkObject *caller, unsigned long eventId, void *callData)
  {
    vtkSmartPointer < vtkRenderWindowInteractor > interactor = dynamic_cast < vtkRenderWindowInteractor * > ( caller ) ;
    // Somehow find the camera
    // Find the renderer first
    vtkSmartPointer < vtkRenderWindow > window = interactor->GetRenderWindow () ;
    vtkSmartPointer < vtkRenderer > renderer = window->GetRenderers ()->GetFirstRenderer () ;
    vtkSmartPointer < vtkCamera > camera = renderer->GetActiveCamera() ;

    if ( this->m_counter > 10 )
      {
       interactor->DestroyTimer ( this->m_timerId ) ;
	std::cout << "destroyed timer" << std::endl ;
      }
    else
      {
	double slice = camera->GetDistance() ;
	slice += 2 ;
	camera->SetDistance ( slice ) ;
        std::cout << "lala" << std::endl;
        std::cout << cVoxel << std::endl;
        //segmentTract(inputsegIterate.GetIndex(), wmtractImage, faImage, pevImage, &iter_no);
	std::cout << slice << " " << this->m_counter << std::endl ;
	interactor->Render() ;
      }
    this->m_counter++ ;

  }

  void SetTimerId ( int id )
  {
    this->m_timerId = id ;
    this->m_counter = 0 ;
  }

private:
  int m_timerId ;
  int m_counter ;
} ;

int main ( int argc, char * argv[] )
{
 // Verify command line arguments
 if( argc < 3 )
    {
      std::cerr << "Usage: " << std::endl ;
      std::cerr << argv[0] << " inputPEVImageFile inputFAImageFile outputWMImageFile" << std::endl ;
      return -1 ;
    }

 const unsigned int nDims = 3 ;
 typedef itk::Vector < double, nDims > vectorType;
 typedef itk::Image < vectorType, nDims > pevImageType ;
 typedef itk::Image < double, nDims > faImageType ;

 // ITK Portion of the code - reading the file in
 // Setup types
 typedef itk::ImageFileReader < pevImageType > ImageReaderType ;
 ImageReaderType::Pointer myReader = ImageReaderType::New() ;
 myReader->SetFileName ( argv[1] ) ;
 pevImageType::Pointer myITKImage = myReader->GetOutput() ;

 // Connect ITK portion to VTK portion
 typedef itk::ImageToVTKImageFilter < pevImageType > ITKToVTKFilterType ;
 ITKToVTKFilterType::Pointer itkToVTKfilter = ITKToVTKFilterType::New() ;
 itkToVTKfilter->SetInput ( myITKImage ) ;
 itkToVTKfilter->Update() ;

  // read the FA iamge in too
  typedef itk::ImageFileReader < faImageType > faImageReaderType ;
  faImageReaderType::Pointer myfaReader = faImageReaderType::New() ;
  myfaReader->SetFileName ( argv[2] ) ;
  faImageType::Pointer myfaITKImage = myfaReader->GetOutput() ;

  typedef itk::ImageToVTKImageFilter < faImageType > faITKToVTKFilterType ;
  faITKToVTKFilterType::Pointer faitkToVTKfilter = faITKToVTKFilterType::New() ;
  faitkToVTKfilter->SetInput ( myfaITKImage ) ;
  faitkToVTKfilter->Update() ;
  faitkToVTKfilter->GetOutput() ;

  // vtkSmartPointer <vtkImageData> vtkImage = connector->GetOutput();
  // VTK Portion of the code - visualization pipeline
  // Mapper
  vtkSmartPointer < vtkImageSliceMapper > imageMapper = vtkSmartPointer < vtkImageSliceMapper > ::New() ;
  imageMapper->SetInputData ( itkToVTKfilter->GetOutput() ) ;
  imageMapper->SetOrientationToX () ;
  imageMapper->SetSliceNumber ( 55 ) ;
  std::cout << "default for atfocalpoint: " << imageMapper->GetSliceAtFocalPoint () << std::endl ;
  std::cout << "default for faces camera: " << imageMapper->GetSliceFacesCamera () << std::endl ;
  imageMapper->SliceAtFocalPointOn () ;
  imageMapper->SliceFacesCameraOn () ;

  // Image property //pk
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
  //renderer->SetBackground(1 ,1, 1) ; //pk

  // Get the camera so we can position it better
  vtkSmartPointer < vtkCamera > camera = renderer->GetActiveCamera() ;

  double position[3],  imageCenter[3] ;
  itkToVTKfilter->GetOutput()->GetCenter ( imageCenter ) ;
  position[0] = imageCenter[0] ;
  position[1] = imageCenter[1] ;
  position[2] = -160 ;
  std::cout << "Image center: " << imageCenter[0] << " " << imageCenter[1] << " " << imageCenter[2] << std::endl ;
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
  std::cout << "Parallel scale: " << camera->GetParallelScale() << std::endl ;
  std::cout << imageDims[0] << " " << imageDims[1] << " " << imageDims[2] << std::endl ;
  camera->SetParallelScale ( imageDims[2] * 2) ;

  // Set up window
  vtkSmartPointer < vtkRenderWindow > window = vtkSmartPointer < vtkRenderWindow >::New() ;
  window->AddRenderer ( renderer ) ;
  window->SetSize ( 500, 500 ) ;

  // Create the interactor
  vtkSmartPointer < vtkRenderWindowInteractor > interactor = vtkSmartPointer < vtkRenderWindowInteractor >::New() ;
  interactor->SetRenderWindow ( window ) ;

  // create a point picker ?

  vtkSmartPointer < vtkInteractorStyleImage > style = vtkSmartPointer < vtkInteractorStyleImage >::New() ;
  // style->SetInteractionModeToImage3D() ; //pk
  style->SetInteractionModeToImageSlicing() ;

  interactor->SetInteractorStyle ( style ) ;
  interactor->Initialize() ;

  // Voxel in corpus collsum
  pevImageType::IndexType cVoxel;
  cVoxel[0] = 69 ;
  cVoxel[1] = 47 ;
  cVoxel[2] = 45 ;
  std::cout << "C Voxel" << cVoxel << std::endl;

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
  //vtkSmartPointer < vtkRenderWindow > renderWindow = vtkSmartPointer < vtkRenderWindow >::New() ;
  //renderWindow->AddRenderer ( renderer ) ;
  //vtkSmartPointer < vtkRenderWindowInteractor > interactor = vtkSmartPointer < vtkRenderWindowInteractor >::New() ;
  //interactor->SetRenderWindow ( renderWindow )

  interactor->CreateRepeatingTimer( 300 ); //pk
  vtkSmartPointer < CustomTimerCallback > myCallback = vtkSmartPointer < CustomTimerCallback >::New() ;
  int timerId = interactor->AddObserver ( vtkCommand::TimerEvent, myCallback, 0 ) ;
  myCallback->SetTimerId ( timerId ) ;

  // Run
  interactor->Start() ;

 return 0;
}