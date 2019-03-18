#include "geometrical_verification.hpp"

GeometricalVerification::GeometricalVerification(int device, bool extended)
    :device_(device), dim_(extended ? 128 : 64)
{
    WarmupOnce();
}

std::pair<std::vector<float>, std::vector<cv::KeyPoint> > 
GeometricalVerification::GetSurfDescriptorsAndKeyPoints(const cv::Mat& img)
{
    std::vector<float> desc;
    std::vector< cv::KeyPoint > keypoints;
    if(img.empty()) return std::make_pair(desc, keypoints);

    cv::cuda::setDevice(device_);

    cv::Mat gray = img;
    if(img.channels() != 1)
    {
        cv::cvtColor(gray, gray, CV_BGR2GRAY);
    }
    //cv::resize(gray, gray, cv::Size(gray.cols/2, gray.rows/2), cv::INTER_NEAREST);
    // Copy the image into GPU memory
    cuda::GpuMat img_Gpu(gray);

    // - the time moving data between GPU and CPU is added
    cuda::GpuMat keypoints_Gpu; // keypoints
    cuda::GpuMat descriptors_Gpu; // descriptors (features)

    //-- Steps 1 + 2, detect the keypoints and compute descriptors, both in one method
    static const int minHessian = 100, nOctaves = 4, nOctaveLayers = 2;

    cuda::SURF_CUDA surf(minHessian, nOctaves, nOctaveLayers, dim_ == 128);
    surf(img_Gpu, cuda::GpuMat(), keypoints_Gpu, descriptors_Gpu);

    // Downloading Descriptor  Gpu -> Cpu
    surf.downloadDescriptors(descriptors_Gpu,  desc);

    // Downloading KeyPoints   Gpu -> Cpu
    surf.downloadKeypoints(keypoints_Gpu, keypoints);

    // Release resource
    surf.releaseMemory();
    img_Gpu.release();
    keypoints_Gpu.release();
    descriptors_Gpu.release();

    return std::make_pair(std::move(desc), std::move(keypoints));
}

void GeometricalVerification::WarmupOnce()
{
    // create a dummy image
    cv::Mat img(cv::Size(500, 500), CV_8UC1, cv::Scalar(255));
    cv::rectangle(img, cv::Rect(20, 20, 150, 150), cv::Scalar(0), 3);
    cv::rectangle(img, cv::Rect(40, 20, 140, 150), cv::Scalar(0), 4);
    cv::rectangle(img, cv::Rect(60, 60, 100, 100), cv::Scalar(0), 2);
    cv::rectangle(img, cv::Rect(70, 90, 150, 120), cv::Scalar(0), 4);
    cv::rectangle(img, cv::Rect(120, 120, 50, 40), cv::Scalar(0), 3);

    auto desc_kpts = std::move(GetSurfDescriptorsAndKeyPoints(img));
    std::cout << " - Warm up, surf feat dim : " << std::get<0>(desc_kpts).size() << std::endl;
    //for(int i = 0; i < 100; ++i)
    //    std::cout << feat[i] << " ";
    //std::cout << std::endl;
}


GeometricalVerification::HashImage GeometricalVerification::ConvertToCascadeHash(const cv::Mat& frame)
{
    //std::pair<std::vector<float>, std::vector<cv::KeyPoint> > 
    auto surf_data = GetSurfDescriptorsAndKeyPoints(frame);
    return 0;
}
