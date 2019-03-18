#ifndef utils_hpp__
#define utils_hpp__
/// Description:
/// Define some publicly used functions or tools.
///////////////////////////////////////////////////////////////////////

#include <atomic>
#include <chrono>
#include <fstream>
#include <string>
#include <thread>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudaarithm.hpp"
#include "opencv2/cudafeatures2d.hpp"
#include "opencv2/xfeatures2d/cuda.hpp"
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <sys/sem.h>


using namespace std;
using namespace chrono;
using namespace cv;

class Parameter
{
public:
	/// add params to map
	void SetParam(const std::string& key, const std::string& val);
	bool HasKey(const std::string& key) { return param_mp_.count(key) > 0; }
	std::string GetValof(const std::string& key) { return param_mp_[key]; }
	void PrintKeys();

private:
	/// preserve params
	std::map<std::string, std::string> param_mp_;
};

class Utils
{
public:
	/// params parsing
	static void ParseConfig(const std::string& config_file);
	template<class T>
	static T GetParamsOf(const std::string& key);

    static double CalcStd(vector<double> &time_vec);    
    static void printMemoryUsage();
	// String is split into arrays by the specified separator
	static void Split(const std::string& str, std::vector<string>& intArr, char c);
	static std::string& Trim(std::string &s); 
    // AVX euclidean distance 
    static float EuclDist(const std::vector<float>& fea_1, const std::vector<float>& fea_2);
    static float get_dist2_avx256_loop_unroll(const float* fea1, const float* fea2, int dim);
private:
	template<class T>
	static T StrTo(const std::string& str);


private:
	static Parameter param_;
};

#endif//utils_hpp__
