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
typedef itk::DiffusionTensor3D < double > DiffTensorType ;
typedef itk::Image < DiffTensorType , nDims > ImageType ;
typedef itk::Vector <double, nDims> VectorType;
typedef itk::Image <VectorType, nDims> PAImageType ;
typedef itk::Image < double, nDims> FAImageType ;
typedef itk::Image < double, nDims> TrackImageType ;
typedef itk::Image < double, nDims> SegmentedImageType ;


int RecursiveTrack(PAImageType::Pointer myPAImage, FAImageType::Pointer myFAImage, TrackImageType::Pointer myTrackImage, ImageType::IndexType thisloc, double delta, int iter)
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
  //std::cout << iter << std::endl;
  iter++;

  // Get next forward and back pixel
  ImageType::IndexType newlocplus = thisloc ;
  ImageType::IndexType newlocneg = thisloc ;
  
  VectorType thisVector ;
  thisVector = myPAImage->GetPixel(thisloc);
  newlocplus [0] = round(thisVector[0]*delta + thisloc[0]);
  newlocplus [1] = round(thisVector[1]*delta + thisloc[1]);
  newlocplus [2] = round(thisVector[2]*delta + thisloc[2]);
  RecursiveTrack(myPAImage, myFAImage, myTrackImage, newlocplus, delta, iter);
  
  newlocneg [0] = round(-thisVector[0]*delta + thisloc[0]);
  newlocneg [1] = round(-thisVector[1]*delta + thisloc[1]);
  newlocneg [2] = round(-thisVector[2]*delta + thisloc[2]);
  RecursiveTrack(myPAImage, myFAImage, myTrackImage, newlocneg , delta, iter);

  return 0;
}


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
  typedef itk::ImageFileReader < ImageType > ImageReaderType1 ;
  ImageReaderType1 ::Pointer myReader1 = ImageReaderType1 ::New() ;  
  myReader1->SetFileName ( argv[1] ) ;
  myReader1->Update();   // go read
  ImageType::Pointer myImage = myReader1->GetOutput();

  typedef itk::ImageFileReader < SegmentedImageType > ImageReaderType2 ;
  ImageReaderType2 ::Pointer myReader2 = ImageReaderType2::New() ;  
  myReader2 ->SetFileName ( argv[2] ) ;
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
  ImageType::IndexType seed ;
  //seed[0] = 73; seed[1] = 69; seed[2] = 32;
  seed[0] = 73; seed[1] = 89; seed[2] = 38;

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

  RecursiveTrack(myPAImage, myFAImageFilter->GetOutput(), myTrackImage, thisloc, delta, iter);

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

  while (!segmentIterator.IsAtEnd() )
  {
   if (segmentIterator.Value() == 1.0)
   {
      int iter = 0;
      RecursiveTrack(myPAImage, myFAImageFilter->GetOutput(), myCCTrackImage , segmentIterator.GetIndex(), delta, iter);
   }
   
   ++segmentIterator ;
  }

   
  //--- Write Result Files ---//
  typedef itk::ImageFileWriter < PAImageType> ImageWriterType1 ;   
  ImageWriterType1 ::Pointer myWriter1 = ImageWriterType1::New();   
  myWriter1->SetFileName( argv[3] );   
  myWriter1->SetInput(myPAImage);  
  myWriter1->Update();   

  typedef itk::ImageFileWriter < FAImageType> ImageWriterType2 ;   
  ImageWriterType2 ::Pointer myWriter2 = ImageWriterType2::New();   
  myWriter2->SetFileName( argv[4] );   
  myWriter2->SetInput(myFAImageFilter->GetOutput() );  
  myWriter2->Update();   

  typedef itk::ImageFileWriter < TrackImageType> ImageWriterType3 ;   
  ImageWriterType3 ::Pointer myWriter3 = ImageWriterType3::New();   
  myWriter3->SetFileName( argv[5] );   
  myWriter3->SetInput(myTrackImage );  
  myWriter3->Update();   

  typedef itk::ImageFileWriter <SegmentedImageType> ImageWriterType4;
  ImageWriterType4 ::Pointer myWriter4 = ImageWriterType4::New();   
  myWriter4->SetFileName( argv[6] );   
  myWriter4->SetInput(myCCTrackImage );  
  myWriter4->Update();   
  

  // Done.
  return 0 ;
}

