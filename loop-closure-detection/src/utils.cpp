#include "utils.hpp"
#include <immintrin.h>
#include <x86intrin.h>

void DoNotCallThisFunction()
{
	const std::string key;
	Utils::GetParamsOf<int>(key);
	Utils::GetParamsOf<float>(key);
	Utils::GetParamsOf<std::string>(key);
	Utils::GetParamsOf<std::string>(key);
}

Parameter Utils::param_;

void Parameter::SetParam(const std::string& key, const std::string& val)
{
	if(param_mp_.count(key) == 0)
		param_mp_[key] = val;		
}

void Parameter::PrintKeys()
{
	std::cout << " - Params now have keys:";
	for(auto& key : param_mp_)
		std::cout << key.first << " ";
	std::cout << std::endl;
}

void Utils::printMemoryUsage() {
  //rusage usage;
  //getrusage(RUSAGE_SELF, &usage);
  //cout << "MEMORY maxrss: " << usage.ru_maxrss << endl;
  //cout << "MEMORY ixrss:  " << usage.ru_ixrss << endl;
  //cout << "MEMORY idrss:  " << usage.ru_idrss << endl;
  //cout << "MEMORY isrss:  " << usage.ru_isrss << endl;

    char buf[30];
    snprintf(buf, 30, "/proc/%u/statm", (unsigned)getpid());
    FILE* pf = fopen(buf, "r");
    if (pf) {
        unsigned size; //       total program size
      //unsigned resident;//   resident set size
      //unsigned share;//      shared pages
      //unsigned text;//       text (code)
      //unsigned lib;//        library
      //unsigned data;//       data/stack
      //unsigned dt;//         dirty pages (unused in Linux 2.6)
        fscanf(pf, "%u" /* %u %u %u %u %u"*/, &size/*, &resident, &share, &text, &lib, &data*/);
        cout << (size / 1024.0) << " MB mem used\n";
        //DOMSGCAT(MSTATS, std::setprecision(4) << size / (1024.0) << "MB mem used");
    }
    fclose(pf);

}

double Utils::CalcStd(vector<double> &time_vec)
{
    double sum = std::accumulate(std::begin(time_vec), std::end(time_vec), 0.0);
    double mean =  sum / time_vec.size();
 
    double accum  = 0.0;
    std::for_each (std::begin(time_vec), std::end(time_vec), [&](const double d) {
        accum  += (d-mean)*(d-mean);
    });
 
    double stdev = sqrt(accum/(time_vec.size()-1));
    return stdev;
}

void Utils::ParseConfig(const std::string& config_file)
{
	std::ifstream ifs(config_file);	

	if(ifs.is_open() == false)
	{
		std::cerr << " - ParseConfig cannot open " << config_file << std::endl;
		return;
	}

	char buffer[1000];
	while (!ifs.eof())
	{
		ifs.getline(buffer, 1000);
		std::string line(buffer);
		if (line.empty()) continue;

		std::vector<std::string> elems;
		Split(line, elems, ':');

		assert(elems.size() == 2 && "Format error in config file!");

		std::string key = Trim(elems[0]);
		std::string val = Trim(elems[1]);

		param_.SetParam(key, val);
	}

	ifs.close();
}

template<class T>
T Utils::GetParamsOf(const std::string& key)
{
	if(param_.HasKey(key))
	{
		return StrTo<T>(param_.GetValof(key));
	}
	else
	{
		std::cout << " - Params do not have this key: " << key << std::endl;
		param_.PrintKeys();	
	}
}


std::string& Utils::Trim(std::string &s) 
{
    if (s.empty()) 
    {
        return s;
    }
 
    s.erase(0,s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
    s.erase(0,s.find_first_not_of("\t"));
    s.erase(s.find_last_not_of("\t") + 1);
    return s;
}
// AVX euclidean distance 
float Utils::EuclDist(const std::vector<float>& fea_1, const std::vector<float>& fea_2)
{
    assert(fea_1.size() == fea_2.isze());
    return get_dist2_avx256_loop_unroll(fea_1.data(), fea_2.data(), fea_1.size());
}

#ifdef __GNUC__
#ifdef __AVX__
float Utils::get_dist2_avx256_loop_unroll(const float* fea1, const float* fea2, int dim)
{
    float s1[8]={0};
    __m256 m1; 
    __m256 m2; 
    __m256 m3; 
    __m256 m4; 
    float *p1 = (float *)fea1;
    float *p2 = (float *)fea2;
    float *p3 = (float *)(fea1+dim/2);
    float *p4 = (float *)(fea2+dim/2);

    int num = int( dim / 16 );
    float dis = 0;
    m2 = _mm256_setzero_ps(); 
    m4 = _mm256_setzero_ps(); 

    for(int i = 0; i < num; i++, p1+=8, p2+=8,p3+=8,p4+=8){
        m1 = _mm256_sub_ps(_mm256_loadu_ps(p1), _mm256_loadu_ps(p2));
        m2 += _mm256_mul_ps(m1, m1);
        m3 = _mm256_sub_ps(_mm256_loadu_ps(p3), _mm256_loadu_ps(p4));
        m4 += _mm256_mul_ps(m3, m3);
    }   

    m2 = _mm256_add_ps(m2, m4);
    _mm256_storeu_ps(s1, m2);
    dis = s1[0] + s1[1] + s1[2] + s1[3] + s1[4] + s1[5] + s1[6] + s1[7];
    if(dim%16!=0){
        for(int j=num*16;j<dim;j++)
        {   
            float diff = fea1[j] - fea2[j]; 
            dis += diff * diff;
        }   
    }   
    //dis = sqrtf(dis);
    return dis;
}
#endif
#endif

// - String is split into arrays by the specified separator
void Utils::Split(const std::string& str, std::vector<string>& intArr, char c)
{
    std::string elem;
    for (size_t i = 0; i <= str.size(); ++i)
    {   
        if (i == str.size() && !elem.empty())
            intArr.emplace_back(elem);
        else if (str[i] == c)
        {   
            if (!elem.empty())
                intArr.emplace_back(elem);
            elem.clear();
        }   
        else
            elem += str[i];
    }   
}
template<>
int Utils::StrTo<int>(const std::string& str)
{
		return std::stoi(str);
}

template<>
float Utils::StrTo<float>(const std::string& str)
{
		return std::stof(str);
}

template<>
std::string Utils::StrTo<std::string>(const std::string& str)
{
		return (str);
}


