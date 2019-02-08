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
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cstdlib> // for exit function
using std::cerr;
using std::endl;
using std::ofstream;


using namespace std;
using namespace cv;
using namespace dlib;

typedef std::vector<std::pair<std::string, std::string>> pairvec;
std::vector<matrix<float,0,1>> face_descriptors;
std::vector<std::pair< matrix<float,0,1>, std::string>> encodings_and_names;



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


void read_directory(const std::string& path, pairvec& folders)
{
    DIR* dirp = opendir(path.c_str());
    struct dirent * dp;
    while ((dp = readdir(dirp)) != NULL) 
    {   
    	std::string filename = dp->d_name;  
    	if ((filename != ".") && (filename != "..") && (filename != "..."))
	    	{    
	    		std::string full_path = path + "/"+filename;
	            folders.push_back(make_pair(dp->d_name , full_path ));
            }
        else
        	    continue;
       
    }
    closedir(dirp);
}


// ----------------------------------------------------------------------------------------
 
int main()
{
    pairvec main_folders;
    pairvec imgs;
    std::string main_folder="train_images";
    read_directory(main_folder, main_folders);	
    for (size_t i = 0; i < main_folders.size(); i ++ )
	    {
	    
	        read_directory(main_folders[i].second, imgs);
	        
	    }

    int j = 0;
    for (int i = 0; i < 5; i ++ )
    	{   
    		       
                    
		            try
		            {

			            frontal_face_detector detector = get_frontal_face_detector();
			            shape_predictor sp;
			            deserialize("shape_predictor_68_face_landmarks.dat") >> sp;
					    anet_type net;
					    deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;
					    matrix<rgb_pixel> img;
					    load_image(img , imgs[i].second );
				
					    std::vector<matrix<rgb_pixel>> faces;
					    for (auto face : detector(img))
						    {
						        auto shape = sp(img, face);
						        matrix<rgb_pixel> face_chip;
						        extract_image_chip(img, get_face_chip_details(shape,150,0.25), face_chip); 
						        faces.push_back(move(face_chip));
						        
						    }
						//cout<<imgs[i].first<<" "<<imgs[i].second<<endl;
						face_descriptors= net(faces);
						cout<<"Processing image-"<<++j<<endl;
						//cout<<++j << trans(face_descriptors[0])<<endl;
	                    std::string image_path = imgs[i].second;
	                    encodings_and_names.push_back(make_pair(face_descriptors[0], image_path));


	                    ofstream outdata;
						outdata.open("encodings_and_names.bin", ios::out | ios::binary); // opens the file
					    if( !outdata )
					     { 
						      cerr << "Error: file could not be opened" << endl;
						      exit(1);
					     }
                        std::vector<std::pair< matrix<float,0,1>, std::string>> ::const_iterator i;
					    for(i=encodings_and_names.begin(); i != encodings_and_names.end();++i )
					     {
					          outdata << "[ " <<i->first << "," << i->second <<" ]"<< "\n";
					     }

                        outdata.close();
					
		    	    }

		            catch(...)
		            {
		            	continue;
		            }
		     
        }
   
    //for (int i=0; i<5; ++i)
      //cout<<encodings_and_names[i].first<<" "<<encodings_and_names[i].second<<endl;

    return 0;

}




 
