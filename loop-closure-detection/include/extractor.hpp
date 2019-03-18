#ifndef extractor_hpp__
#define extractor_hpp__

#define USE_OPENCV

#include "caffe/layers/memory_data_layer.hpp"
#include <caffe/caffe.hpp>
#include "utils.hpp" 

using namespace caffe;

class Extractor
{
public:
	explicit Extractor(const std::string& deploy, const std::string& weight, const int gpu_id);

	std::vector<float> DoInference(const cv::Mat& image);
    void DoInferenceInfo(const cv::Mat& image);

private:
    void WarmupOnce();
private:
	std::shared_ptr<Net<float> > net_ptr_;
    const int gpu_id_ = 0;
};



#endif//extractor_hpp__
