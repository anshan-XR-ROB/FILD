#ifndef lcdengine_hpp__
#define lcdengine_hpp__

#include "utils.hpp"

class Index;
class Extractor;
class GeometricalVerification;

class LCDEngine
{
public:
    LCDEngine();
    void OnInit();
    void Run();
    void OnDestroy();

private:
    std::vector<std::string> 
    ReadFrameAbsPaths(const std::string& namelist_file, const std::string& image_dir);

private:
    std::shared_ptr<Index> hnsw_engine_;
    std::shared_ptr<Extractor> extractor_;
    std::shared_ptr<GeometricalVerification> geom_verif_;

    std::vector<std::string> frame_abspath_vec_;
};


#endif//lcdengine_hpp__
