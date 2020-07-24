#include "extractor.hpp"

Extractor::Extractor(const std::string& deploy, const std::string& weight, const int gpu_id)
    : gpu_id_(gpu_id)
{
	Caffe::set_mode(Caffe::GPU);
	Caffe::SetDevice(gpu_id);
	net_ptr_.reset(new Net<float>(deploy, TEST));
	net_ptr_->CopyTrainedLayersFrom(weight);
    
    WarmupOnce(); 
}

void Extractor::WarmupOnce()
{
    // Dummy image inference
    cv::Mat img(cv::Size(224, 224), CV_8UC3, cv::Scalar(0, 0, 0));
    std::cout << " - Start Warm up " << std::endl; 
    std::vector<float> feat = std::move(DoInference(img));
    std::cout << " - Warm up, CNN feat dim : " << feat.size() << std::endl;
    assert(int(feat.size()) == utilobj.GetParamsOf<int>("cnn_feat_dim"));
    //for(int i = 0; i < 100; ++i)
    //    std::cout << feat[i] << " ";
    //std::cout << std::endl;
}

std::vector<float> Extractor::DoInference(const cv::Mat& image)
{
	Caffe::SetDevice(gpu_id_);
    static caffe::MemoryDataLayer<float> *mem_data_layer = (caffe::MemoryDataLayer<float> *)net_ptr_->layers()[0].get();
    static int input_width = mem_data_layer->width();
    static int input_height = mem_data_layer->height();
    // std::cout << input_width << ", " << input_height << std::endl;

    std::vector<cv::Mat> imgs{image};

    std::vector<int> int_vec;
    std::vector<cv::Mat> sample_v(imgs.size());
    for(int i = 0; i < imgs.size(); ++i)
    {
        cv::resize(imgs[i], sample_v[i], cv::Size(input_width, input_height), INTER_NEAREST);
        int_vec.push_back(i);
    }
    mem_data_layer->set_batch_size(sample_v.size());
    mem_data_layer->AddMatVector(sample_v, int_vec);

    // std::cout << "Forward..." << std::endl;
    net_ptr_->Forward();

    Blob<float>* labels_blob = net_ptr_->output_blobs()[0];
    Blob<float>* result_blob = net_ptr_->output_blobs()[1];
    const float* labels = labels_blob->cpu_data();
    const float* result = result_blob->cpu_data();

    int offset = result_blob->channels() * result_blob->height() * result_blob->width();

    /*
    std::cout << " - result blob shape(NCHW): " << result_blob->num() << ", " << result_blob->channels() << ", " << result_blob->height() << ", " << result_blob->width() << std::endl;;  
    for(auto v : net_ptr_->output_blob_indices()) std::cout << v << ": " << net_ptr_->blob_names()[v] << ", ";
    std::cout << std::endl;
    exit(0);
    */
    std::vector<std::vector<float> > feats;
    feats.resize(labels_blob->count());
    for(int d = 0; d < labels_blob->count(); ++d)
    {
        int indices = labels[d];
        int b_ind = (indices + 0) * offset;
        int e_ind = (indices + 1) * offset;
        //std::cout << " - indices " << indices << ", [" << b_ind << ", " << e_ind << "]" <<std::endl;
        feats[indices].assign(result + b_ind, result + e_ind);
    }

    return feats.back();
}
void Extractor::DoInferenceInfo(const cv::Mat& image)
{
	Caffe::SetDevice(gpu_id_);
    static caffe::MemoryDataLayer<float> *mem_data_layer = (caffe::MemoryDataLayer<float> *)net_ptr_->layers()[0].get();
    static int input_width = mem_data_layer->width();
    static int input_height = mem_data_layer->height();
    // std::cout << input_width << ", " << input_height << std::endl;

    std::vector<cv::Mat> imgs{image};

    std::vector<int> int_vec;
    std::vector<cv::Mat> sample_v(imgs.size());
    for(int i = 0; i < imgs.size(); ++i)
    {
        imgs[i].convertTo(sample_v[i], CV_32F); 
        cv::resize(sample_v[i], sample_v[i], cv::Size(input_width, input_height), INTER_NEAREST);
        int_vec.push_back(i);
    }
    mem_data_layer->set_batch_size(sample_v.size());
    mem_data_layer->AddMatVector(sample_v, int_vec);

    // std::cout << "Forward..." << std::endl;
    net_ptr_->Forward();

    const std::string layername = "data";
    Blob<float>* result_blob = net_ptr_->blob_by_name(layername).get();

    ///////////////////////////////////////////////////////
    int n = 0, c = 0, h = 0;
    for(int w = 0; w < 10; ++w)
    {
    
        std::cout << result_blob->data_at(n, c, h, w) << " ";
    }
    std::cout << std::endl;
    ///////////////////////////////////////////////////////

    /*
    Blob<float>* labels_blob = net_ptr_->output_blobs()[0];
    Blob<float>* result_blob = net_ptr_->output_blobs()[1];
    const float* labels = labels_blob->cpu_data();
    const float* result = result_blob->cpu_data();

    int offset = result_blob->channels() * result_blob->height() * result_blob->width();

    std::cout << " - result blob shape(NCHW): " << result_blob->num() << ", " << result_blob->channels() << ", " << result_blob->height() << ", " << result_blob->width() << std::endl;;  
    for(auto v : net_ptr_->output_blob_indices()) std::cout << v << ": " << net_ptr_->blob_names()[v] << ", ";
    std::cout << std::endl;
    exit(0);

    std::vector<std::vector<float> > feats;
    feats.resize(labels_blob->count());
    for(int d = 0; d < labels_blob->count(); ++d)
    {
        int indices = labels[d];
        int b_ind = (indices + 0) * offset;
        int e_ind = (indices + 1) * offset;
        //std::cout << " - indices " << indices << ", [" << b_ind << ", " << e_ind << "]" <<std::endl;
        feats[indices].assign(result + b_ind, result + e_ind);
    }
    */
}
