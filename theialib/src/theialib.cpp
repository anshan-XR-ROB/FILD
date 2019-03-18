#include "theialib.hpp"

void TheiaTool::OnInit(const int dim)
{
    cashasher_.reset(new CascadeHasher()); 
    cashasher_->Initialize(dim);
}

std::vector<Eigen::VectorXf> TheiaTool::StdVecToEigenVec(const std::vector<float>& desc)
{
    std::vector<Eigen::VectorXf> res;
    if(desc.size() % dim_ != 0)
    {
        std::cout << "desc.size() % dim != 0" << std::endl;
        return res;
    }
    for(int i = 0; i < desc.size(); i += dim_)
    {
        //res.emplace_back(Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(desc.data() + i, dim));
        res.emplace_back(Eigen::VectorXf::Map((float*)desc.data() + i, dim_));
    }
    return res;
}

std::shared_ptr<HashedImage> TheiaTool::CreateHashedDescriptors(const std::vector<float>& desc)
{
    return cashasher_->CreateHashedSiftDescriptors(StdVecToEigenVec(desc));
}

std::vector<IndexedFeatureMatch> TheiaTool::Match(std::shared_ptr<HashedImage> q, const std::vector<float>& q_desc,
                                                  std::shared_ptr<HashedImage> db, const std::vector<float>& db_desc)
{
    const double lowes_ratio = 0.6;
    std::vector<IndexedFeatureMatch> matches_q_db;
    std::vector<Eigen::VectorXf> q_vec = StdVecToEigenVec(q_desc);
    std::vector<Eigen::VectorXf> db_vec = StdVecToEigenVec(db_desc);
    cashasher_->MatchImages(q, q_vec, db, db_vec, lowes_ratio, &matches_q_db);
    return matches_q_db;
}

std::vector<IndexedFeatureMatch> TheiaTool::Match(std::shared_ptr<HashedImage> q, std::shared_ptr<HashedImage> db)
{ 
    //auto start = system_clock::now();
    const double lowes_ratio = 0.7;
    std::vector<IndexedFeatureMatch> matches_q_db;
    cashasher_->MatchImages(q, db, lowes_ratio, &matches_q_db); 
    
    //cross check
    /*std::vector<IndexedFeatureMatch> matches_db_q;
    cashasher_->MatchImages(db, q, lowes_ratio, &matches_db_q); 
    std::vector<IndexedFeatureMatch> result;
    for(int i = 0; i < matches_q_db.size(); ++i)
    {
        for(int j = 0; j < matches_db_q.size(); ++j)
        {
            if(matches_q_db[i].feature1_ind == matches_db_q[j].feature2_ind
                    && matches_q_db[i].feature2_ind == matches_db_q[j].feature1_ind
            )
            result.emplace_back(IndexedFeatureMatch(matches_q_db[i].feature1_ind, 
                    matches_q_db[i].feature2_ind, matches_q_db[i].distance));
        }
    } */
    /*auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    float time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
    std::cout << "TheiaTool::Match cost"
        << time
        << " sec " << "Matched:" << matches_q_db.size() << std::endl;
    */
    return matches_q_db;
}


#ifdef THEIA_HAS_THREAD_LOCAL_KEYWORD
thread_local std::mt19937 util_generator;
#else
static std::mt19937 util_generator;
#endif  // THEIA_HAS_THREAD_LOCAL_KEYWORD

RandomNumberGenerator::RandomNumberGenerator() {
    const unsigned seed =
        std::chrono::system_clock::now().time_since_epoch().count();
    util_generator.seed(seed);
}

RandomNumberGenerator::RandomNumberGenerator(const unsigned seed) {
    util_generator.seed(seed);
}

void RandomNumberGenerator::Seed(const unsigned seed) {
    util_generator.seed(seed);
}

// Get a random double between lower and upper (inclusive).
double RandomNumberGenerator::RandDouble(const double lower,
        const double upper) {
    std::uniform_real_distribution<double> distribution(lower, upper);
    return distribution(util_generator);
}

float RandomNumberGenerator::RandFloat(const float lower, const float upper) {
    std::uniform_real_distribution<float> distribution(lower, upper);
    return distribution(util_generator);
}

// Get a random int between lower and upper (inclusive).
int RandomNumberGenerator::RandInt(const int lower, const int upper) {
    std::uniform_int_distribution<int> distribution(lower, upper);
    return distribution(util_generator);
}

// Gaussian Distribution with the corresponding mean and std dev.
double RandomNumberGenerator::RandGaussian(const double mean,
        const double std_dev) {
    std::normal_distribution<double> distribution(mean, std_dev);
    return distribution(util_generator);
}

Eigen::Vector2d RandomNumberGenerator::RandVector2d(const double min,
        const double max) {
    return Eigen::Vector2d(RandDouble(min, max), RandDouble(min, max));
}

Eigen::Vector2d RandomNumberGenerator::RandVector2d() {
    return RandVector2d(-1.0, 1.0);
}

Eigen::Vector3d RandomNumberGenerator::RandVector3d(const double min,
        const double max) {
    return Eigen::Vector3d(RandDouble(min, max),
            RandDouble(min, max),
            RandDouble(min, max));
}

Eigen::Vector3d RandomNumberGenerator::RandVector3d() {
    return RandVector3d(-1.0, 1.0);
}

Eigen::Vector4d RandomNumberGenerator::RandVector4d(const double min,
        const double max) {
    return Eigen::Vector4d(RandDouble(min, max),
            RandDouble(min, max),
            RandDouble(min, max),
            RandDouble(min, max));
}

Eigen::Vector4d RandomNumberGenerator::RandVector4d() {
    return RandVector4d(-1.0, 1.0);
}

