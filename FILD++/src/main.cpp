#include "lcdengine.hpp"


int main()
{


    const std::string config_file = "../resource/config_newcollege_right.txt";
    //const std::string config_file = "../resource/config_malaga.txt";
	//const std::string config_file = "../resource/config_mh05.txt";
	//const std::string config_file = "../resource/config_kitti00.txt";
	//const std::string config_file = "../resource/config_city.txt";
	//const std::string config_file = "../resource/config_StLucia_190809_0845.txt";

    Utils::ParseConfig(config_file);

    LCDEngine lcdengine;
    lcdengine.Run();
	return 0;
}
