#ifndef theialib_hpp__
#define theialib_hpp__

//#define EIGEN_USE_MKL_ALL
//#define EIGEN_NO_DEBUG 
#include <memory>
#include <Eigen/Core>
#include <stdint.h>
#include <bitset>
#include <vector>
#include <chrono>  // NOLINT
#include <random>
#include <glog/logging.h>
#include <algorithm>
#include <cmath>
#include <utility>

#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"

using namespace std;
using namespace chrono;
struct HashedImage;
class CascadeHasher;

struct IndexedFeatureMatch {
    IndexedFeatureMatch() {}
    IndexedFeatureMatch(int f1_ind, int f2_ind, float dist)
        : feature1_ind(f1_ind), feature2_ind(f2_ind), distance(dist) {}

    // Index of the feature in the first image.
    int feature1_ind;
    // Index of the feature in the second image.
    int feature2_ind;
    // Distance between the two features.
    float distance;
};

class TheiaTool
{
public:
    TheiaTool(const int dim) :dim_(dim) { OnInit(dim); }    
    void OnInit(const int dim);
    std::vector<IndexedFeatureMatch> Match(std::shared_ptr<HashedImage> q, std::shared_ptr<HashedImage> db);
    std::vector<IndexedFeatureMatch> Match(std::shared_ptr<HashedImage> q, const std::vector<float>& q_desc, 
                                           std::shared_ptr<HashedImage> db, const std::vector<float>& db_desc);
    std::shared_ptr<HashedImage> CreateHashedDescriptors(const std::vector<float>& desc);
private:
    
    template<typename _Tp>
    std::vector<_Tp> ConvertMat2Vector(const cv::Mat &mat)
    {
        return (std::vector<_Tp>)(mat.reshape(1, 1));//通道数不变，按行转为一行
    }

    std::vector<Eigen::VectorXf> StdVecToEigenVec(const std::vector<float>& desc);
    std::shared_ptr<CascadeHasher> cashasher_;
    const int dim_ = 128;
};

// L2 distance for euclidean features.
// Squared Euclidean distance functor. We let Eigen handle the SSE optimization.
// NOTE: This assumes that each vector has a unit norm:
//  ||x - y||^2 = ||x||^2 + ||y||^2 - 2*||x^t * y|| = 2 - 2 * x.dot(y).
struct L2 {
    typedef float DistanceType;
    typedef Eigen::VectorXf DescriptorType;

    DistanceType operator()(const Eigen::VectorXf& descriptor_a,
            const Eigen::VectorXf& descriptor_b) const {
        DCHECK_EQ(descriptor_a.size(), descriptor_b.size());
        return (descriptor_a - descriptor_b).squaredNorm();
    }
};

// Used for sorting a vector of the feature matches.
inline bool CompareFeaturesByDistance(const IndexedFeatureMatch& feature1,
        const IndexedFeatureMatch& feature2) {
    return feature1.distance < feature2.distance;
}

// A wrapper around the c++11 random generator utilities. This allows for a
// thread-safe random number generator that may be easily instantiated and
// passed around as an object.
class RandomNumberGenerator {
    public:
        // Creates the random number generator using the current time as the seed.
        RandomNumberGenerator();

        // Creates the random number generator using the given seed.
        explicit RandomNumberGenerator(const unsigned seed);

        // Seeds the random number generator with the given value.
        void Seed(const unsigned seed);

        // Get a random double between lower and upper (inclusive).
        double RandDouble(const double lower, const double upper);

        // Get a random float between lower and upper (inclusive).
        float RandFloat(const float lower, const float upper);

        // Get a random double between lower and upper (inclusive).
        int RandInt(const int lower, const int upper);

        // Generate a number drawn from a gaussian distribution.
        double RandGaussian(const double mean, const double std_dev);

        // Return eigen types with random initialization. These are just convenience
        // methods. Methods without min and max assign random values between -1 and 1
        // just like the Eigen::Random function.
        Eigen::Vector2d RandVector2d(const double min, const double max);
        Eigen::Vector2d RandVector2d();
        Eigen::Vector3d RandVector3d(const double min, const double max);
        Eigen::Vector3d RandVector3d();
        Eigen::Vector4d RandVector4d(const double min, const double max);
        Eigen::Vector4d RandVector4d();
        // Sets an Eigen type with random values between -1.0 and 1.0. This is meant
        // to replace the Eigen::Random() functionality.
        template <int RowsT, int ColsT>
            void SetRandom(Eigen::Matrix<double, RowsT, ColsT>* b) {
                double* data = b->data();
                for (int i = 0; i < b->size(); i++) {
                    data[i] = RandDouble(-1.0, 1.0);
                }
            }

        template <int RowsT, int ColsT>
            void SetRandom(Eigen::Matrix<float, RowsT, ColsT>* b) {
                float* data = b->data();
                for (int i = 0; i < b->size(); i++) {
                    data[i] = RandFloat(-1.0, 1.0);
                }
            }
};

struct IndexedFeatureMatch;
typedef std::vector<int> Bucket;

// The number of dimensions of the Hash code.
static const int kHashCodeSize = 256;
// The number of bucket bits.
static const int kNumBucketBits = 10;
// The number of bucket groups.
static const int kNumBucketGroups = 6;
// The number of buckets in each group.
static const int kNumBucketsPerGroup = 1 << kNumBucketBits;

struct HashedSiftDescriptor {
    // Hash code generated by the primary hashing function.
    std::bitset<kHashCodeSize> hash_code;
    // Each bucket_ids[x] = y means the descriptor belongs to bucket y in bucket
    // group x.
    std::vector<uint16_t> bucket_ids;
};

struct HashedImage {
    HashedImage() {}

    // The mean of all descriptors (used for hashing).
    Eigen::VectorXf mean_descriptor;

    // The hash information.
    std::vector<HashedSiftDescriptor> hashed_desc;

    // buckets[bucket_group][bucket_id] = bucket (container of sift ids).
    std::vector<std::vector<Bucket> > buckets;

    // Chegf: The descriptors num
    const size_t NumOfDesc() const { return hashed_desc.size(); }
};

// This hasher will hash SIFT descriptors with a two-step hashing system. The
// first generates a hash code and the second determines which buckets the
// descriptors belong to. Descriptors in the same bucket are likely to be good
// matches.
//
// Implementation is based on the paper "Fast and Accurate Image Matching with
// Cascade Hashing for 3D Reconstruction" by Cheng et al (CVPR 2014). When using
// this class we ask that you please cite this paper.
class CascadeHasher {
    public:
        CascadeHasher() : rng_(std::make_shared<RandomNumberGenerator>()) {}
        CascadeHasher(std::shared_ptr<RandomNumberGenerator> rng) : rng_(rng) {}

        // Creates the hashing projections. This must be called before using the
        // cascade hasher.
        bool Initialize(const int num_dimensions_of_descriptor);

        // Creates the hash codes for the sift descriptors and returns the hashed
        // information.
        std::shared_ptr<HashedImage> CreateHashedSiftDescriptors(
                const std::vector<Eigen::VectorXf>& sift_desc) const;

        // Matches images with a fast matching scheme based on the hash codes
        // previously generated.
        int MatchImages(std::shared_ptr<HashedImage> hashed_desc1,
                std::vector<Eigen::VectorXf>& descriptors1,
                std::shared_ptr<HashedImage> hashed_desc2,
                std::vector<Eigen::VectorXf>& descriptors2,
                const double lowes_ratio,
                std::vector<IndexedFeatureMatch>* matches) const;
        
        // Matches images with a fast matching scheme based on the hash codes
        int MatchImages(
                std::shared_ptr<HashedImage> hashed_desc1,
                std::shared_ptr<HashedImage> hashed_desc2,
                const double lowes_ratio,
                std::vector<IndexedFeatureMatch>* matches) const;

    private:
        std::shared_ptr<RandomNumberGenerator> rng_;

        // Creates the hash code for each descriptor and determines which buckets each
        // descriptor belongs to.
        void CreateHashedDescriptors(const std::vector<Eigen::VectorXf>& sift_desc,
                HashedImage* hashed_image) const;

        // Builds the buckets for an image based on the bucket ids and groups of the
        // sift descriptors.
        void BuildBuckets(HashedImage* hashed_image) const;

        // Number of dimensions of the descriptors.
        int num_dimensions_of_descriptor_;

        // Projection matrix of the primary hashing function.
        Eigen::MatrixXf primary_hash_projection_;

        // Projection matrices of the secondary hashing function.
        Eigen::MatrixXf secondary_hash_projection_[kNumBucketGroups];
};



#endif//theialib_hpp__
