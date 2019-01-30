
//It is to detect faces in video captured using webcam.

// Include required header files from OpenCV directory 
#include "opencv2/objdetect.hpp" 
#include "opencv2/highgui.hpp" 
#include "opencv2/imgproc.hpp" 
#include <iostream> 

using namespace std; 
using namespace cv; 

// Function for Face Detection 
void detectAndDraw( Mat& img, CascadeClassifier& cascade ); 
string cascadeName;

int main( int argc, const char** argv ) 
{ 
	// VideoCapture class for playing video for which faces to be detected 
	VideoCapture capture; 
	Mat frame, image; 

	// PreDefined trained XML classifiers with facial features 
	CascadeClassifier cascade; 
	
	// Change path before execution 
	cascade.load( "haarcascade_frontalface_default.xml" ) ; 

	capture.open(-1); 
	if( capture.isOpened() ) 
	{ 
		// Capture frames from video and detect faces 
		cout << "Face Detection Started...." << endl; 
		while(1) 
		{ 
			capture >> frame; 
			if( frame.empty() ) 
				break; 
		 
			detectAndDraw( frame, cascade ); 
			char c = (char)waitKey(10); 
          
            // Press q to exit from window 
            if( c == 27 || c == 'q' || c == 'Q' )  
                break; 
		} 
	} 
	else
		cout<<"Could not Open Camera"; 
	return 0; 
} 

void detectAndDraw( Mat& img, CascadeClassifier& cascade)



{  //It is to draw rectangle around the detected faces.
 try {
          
		  vector<Rect> faces;
		  Mat frame_gray;

		  cvtColor( img, frame_gray, CV_BGR2GRAY );
		  equalizeHist( frame_gray, frame_gray );

		  //-- Detect faces
		  cascade.detectMultiScale( frame_gray, faces, 1.3,5);

		  for( size_t i = 0; i < faces.size(); i++ )
		  {
		     rectangle(img, Point (faces[i].x, faces[i].y),Point (faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar(255,0,0), 3);
		    
		  }
		
	        
	    }
 catch(...)
              {
                  cout<<"\nException occur.";
              }

	// Show Processed Image with detected faces 
	imshow( "Face Detection", img ); 
} 
