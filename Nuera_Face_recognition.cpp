#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#include <dlib/opencv.h>
#include <dlib/image_io.h>
#include <sys/types.h>
#include <dirent.h>
#include <utility> 
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include "opencv2/objdetect/objdetect.hpp"
#include <opencv2/core/core.hpp>
#include "opencv2/imgproc.hpp" 
#include "opencv2/videoio.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib> // for exit function
#include <cstddef> 

using namespace std;
using namespace cv;
using namespace dlib;
using std::cerr;
using namespace std;



template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N,BN,1,tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2,2,2,2,skip1<tag2<block<N,BN,2,tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET> 
using block  = BN<con<N,3,3,1,1,relu<BN<con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block,N,affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block,N,affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            max_pool<3,3,2,2,relu<affine<con<32,7,7,2,2,
                            input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


struct encodings
{  
	matrix<float,0,1> encoding;
    std::string image_path;

};
typedef encodings vec[180];
vec dataset;
typedef std::string str;


void fetch_data()
{
	ifstream encodings, names;
	encodings.open("encodings_180.bin", ios::out | ios::binary); // opens the file
	names.open("names_180.bin", ios::out | ios::binary);
	if( !encodings || !names )
	{ 
	      cerr << "Error: file could not be opened" << endl;
	      exit(1);
	}
	else
	{   
		for ( int i = 0; i < 180 ; i++)
	        {   
	            encodings >> dataset[i].encoding;
	            encodings.ignore();
	            getline(names,dataset[i].image_path);
	        } 
	    names.close();
	    encodings.close();
	}
	cout<<"Data loaded!\n";
}

Mat preprocess(Mat& frame)
{
	Mat frame_gray, frame_hist;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist(frame_gray,  frame_hist);
    return frame_hist;
}


std::vector<matrix<float,0,1>> encode_face(Mat& cropped_image)
{   
	anet_type net;
    deserialize("/home/root/Face-recognition/dlib_face_recognition_resnet_model_v1.dat")>> net;
	matrix<rgb_pixel> img;
    assign_image(img, cv_image<bgr_pixel>(cropped_image));
	std::vector<matrix<rgb_pixel>> faces;
	faces.push_back(img);
	std::vector<matrix<float,0,1>> face_descriptors;
	try
	{
       face_descriptors = net(faces);
       //cout<<trans(face_descriptors[0])<<endl;
       return face_descriptors;
    }
	catch(...)  
	{ 
		cout<<"Exception"<<endl;
		return face_descriptors;
	}
	
	
	
}
 
str get_foldername(str& image_path)
{   
    size_t pos = 0;
	std::vector<str> parts;
	str delimiter = "/";
	try
	{
		while ((pos = image_path.find(delimiter)) != std::string::npos) 
		{
		    str part = image_path.substr(0, pos);
		    parts.push_back(part);
		    image_path.erase(0, pos + delimiter.length());
		}
		return parts[1];
	}

	catch(...)
	{   cout<<"Exception"<<endl;
		return "   ";
	}
	
}

str recognize_faces(std::vector<matrix<float,0,1>>& face_descriptors)
{  

    str name;
	std::vector<std::pair<float, str>> distances;
    for (size_t i = 0; i < face_descriptors.size(); ++i)
    {   
    	cout<<"\n";
    	for (auto each : dataset)
        {   
        	float distance = length(face_descriptors[i] - each.encoding);
	        if ( distance < 0.55)
	         	{
                    distances.push_back(make_pair(distance,each.image_path) );   
	         	}
	    }
    }
    
    if (distances.empty())  name = "  ";
    else if(distances.size() == 1) 
    {   
        cout<<distances[0].first<<"----"<<distances[0].second<<endl;
    	name = get_foldername(distances[0].second);
    }
    else 
    {   
    	sort(distances.begin(), distances.end());
    	for (auto each : distances)  
    	{  
    	   cout<<each.first<<"----"<<each.second<<endl; //print the sorted encodings on screen
    	}
	    float min_diff = 0.001;
	    float difference = distances[1].first - distances[0].first;
	    if (difference > min_diff) name = get_foldername(distances[0].second);
	    else if (difference < min_diff)  
	    {  
	       str foldername_1 = get_foldername(distances[0].second);
	       str foldername_2 = get_foldername(distances[1].second); 
	       if (!( foldername_1.compare(foldername_2))) name = foldername_1;
           else name = "  ";
	    }
    }
    return name;

}

void detect_draw(Mat& processed_frame,Mat& frame, CascadeClassifier& cascade)
{
	std::vector<Rect> faces;
	std::vector<matrix<float,0,1>> face_descriptors;
	str name;
	cascade.detectMultiScale(processed_frame, faces, 1.3,5);
	if (faces.size() != 0)
	{
      for( size_t i = 0; i < faces.size(); i++ )
	  {   
	  	  int x = faces[i].x;
	      int y = faces[i].y;
	  	  int w = faces[i].width;
	  	  int h = faces[i].height;
	      cv::rectangle(frame, Point (x, y), Point (x + w, y + h), Scalar(255,0,0), 3);
	      Mat cropped_image = frame(Rect(x, y, w, h));
	      resize(cropped_image, cropped_image, cv::Size(150,150));
	      /*imshow("cropped_image", cropped_image);   
	      waitKey(0);
	      destroyAllWindows();*/
          face_descriptors = encode_face(cropped_image);
          name = recognize_faces(face_descriptors);
          cout<<name<<endl;
	      cv::rectangle(frame, Point (x, y - 35),Point (x + w, y), Scalar(255, 0, 0), cv::FILLED);
          cv::putText(frame, name, Point (x + 6, y - 6),cv::FONT_HERSHEY_DUPLEX ,0.5, Scalar(255, 255, 255), 1);
	  }
	}

	//*****Enter code for processing the output i.e frame******


    cv::imshow( "SFR", frame );
    cv::waitKey(0);
	cv::destroyAllWindows();
}


void face_recognition(Mat& frame, CascadeClassifier& cascade)
{
   
	Mat processed_frame = preprocess(frame);
	detect_draw( processed_frame,frame, cascade); 
	
}

int main()
{   
	/* Initialization sequence */
	CascadeClassifier cascade; 
    cascade.load( "/home/root/Face-recognition/haarcascade_frontalface_default.xml" ); 
	fetch_data();
	Mat image = imread("/home/root/Face-recognition/3.jpeg", IMREAD_COLOR);  
	
	/* Face Recognition Algo */
	face_recognition(image , cascade);
	return 0;
}
