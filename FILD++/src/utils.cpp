#include "utils.hpp"
//#include <immintrin.h>
//#include <x86intrin.h>
#include <numeric>

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


