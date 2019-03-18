#ifndef geometrical_verification_hpp__
#define geometrical_verification_hpp__

#include "utils.hpp"

class GeometricalVerification
{
public:
    GeometricalVerification(int device, bool extended = true);

    std::pair<std::vector<float>, std::vector<cv::KeyPoint> > 
            GetSurfDescriptorsAndKeyPoints(const cv::Mat& img);

    typedef int HashImage;
    HashImage ConvertToCascadeHash(const cv::Mat& frame);

private:
    void WarmupOnce();
private:
    const int dim_ = 128;
    const int device_ = 0;
};

#endif// geometrical_verification_hpp__
