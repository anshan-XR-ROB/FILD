#include "lcdengine.hpp"


int main()
{
#if 0
	Utils utilobj;	
	
	const std::string config_file = "../resource/config.txt";

	utilobj.ParseConfig(config_file);

	std::string image_name_list = utilobj.GetParamsOf<std::string>("image_name_list");	
	int frame_rate = utilobj.GetParamsOf<int>("frame_rate");	
	float similarity_threhold = utilobj.GetParamsOf<float>("similarity_threhold");	

	std::cout << "image_name_list: " << image_name_list << std::endl;
	std::cout << "frame_rate: " << frame_rate << std::endl;
	std::cout << "similarity_threhold: " << similarity_threhold << std::endl;
#endif 

#if 0
    const std::string deploy = "../resource/model_res152/deploy_resnet152_places365.prototxt";
    const std::string weight = "../resource/model_res152/resnet152_places365.caffemodel";
    const int gpu_id = 1;

    Extractor extractor(deploy, weight, gpu_id);
#endif
	const std::string config_file = "../resource/config_newcollege.txt";
    Utils::ParseConfig(config_file);

    LCDEngine lcdengine;
    lcdengine.Run();
	return 0;
}
