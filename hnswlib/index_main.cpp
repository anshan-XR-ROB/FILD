#include <random>
#include <algorithm>
#include <set>
#include "omp.h"
#include <stdio.h>
#include <fstream>
#include <thread>
#include <iostream>
#include <map>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <stdlib.h>
#include <string.h>

#include "hnswlib.hpp"

using namespace std;
using namespace chrono;
using namespace hnswlib;

#ifdef WIN32
#include <direct.h>
#include "dirent_win.h"
#define mkdir(a) _mkdir(a)
#else
#include <dirent.h>
#include <sys/stat.h>
#define mkdir(a) mkdir(a, 0755)
#endif

#define DIM 2048
typedef std::tuple<std::string, std::vector<float>, std::vector<string> > QueryDataType;

std::vector<float> eval(const std::vector<std::string>& top_md5s, const std::vector<std::string>& gt_md5s, const std::vector<int>& cursors);
void loadQueryData(string& file, std::vector<QueryDataType>& db);
void loadTrainData(string& file, std::map<int, vector<float> >& db, std::map<int, std::string>& labelmap); 
int countLines(const char *filename);
inline void showProgressBar(int cnt, int total, int interval = 100);
int main()
{
	int categ = 1;
	std::string rsc = "../../validation-02/fetch_data/";	

	std::string traindb = rsc + "BaseFeats_cate_" + 
			std::to_string(categ) + ".txt";

	// load query feats
	std::string querydb = rsc + "QueryFeats_Cate_" + 
			std::to_string(categ) + "_1k.txt";
	std::vector<QueryDataType> query_data;	
	loadQueryData(querydb, query_data);	

	const std::string path_to_index = std::string("Cate_") + std::to_string(categ) + ".nsw";
	Index indobj("l2", DIM);	

	// load train databse
	std::map<int, vector<float> > train_data;
	std::map<int, std::string> labelmap;
	loadTrainData(traindb, train_data, labelmap);
	
#if 1 
	indobj.CreateNewIndex(train_data.size(), 16, 200);
    std::cout << " - Building index\n";

    auto start = system_clock::now();
	for(auto& pr : train_data)
	{
		indobj.AddItem((void *)pr.second.data(), pr.first);

        std::cout << " - insert label: " << pr.first << std::endl; 
        for(int i = 0; i < 10; ++i)
            std::cout << pr.second[i] << " ";
        std::cout << std::endl;

        std::cout << " - search label: " << pr.first << std::endl; 
        std::vector<float> res = indobj.GetPoint(pr.first); 
        for(int i = 0; i < 10; ++i)
            std::cout << res[i] << " ";
        std::cout << std::endl;

        static int loop = 0;
        if(++loop == 10) exit(0);
        showProgressBar(pr.first, train_data.size(), 100);
	}
	indobj.SaveIndex(path_to_index);
#else
    auto start = system_clock::now();
	indobj.LoadIndex(path_to_index, train_data.size());
#endif

    auto end = system_clock::now();
    auto duration = duration_cast<microseconds>(start - end);
	std::cout << " - build hnsw cost"
			<< double(duration.count()) * microseconds::period::num / microseconds::period::den
			<< " seconds" << std::endl;
	
	// do query and eval
	const int top_n = 5000;
	std::vector<int> cursors{1, 2, 4, 8, 10, 20, 40, 100, 200, 400, 800, 1000, 2000, 3000, 4000, 5000};
	std::vector<float> avg_recalls(cursors.size(), 0.f);
	std::ofstream ofs("result.txt");
	for(size_t i = 0; i != query_data.size(); ++i)
	{
        showProgressBar(i + 1, query_data.size(), 50);
		const QueryDataType& qdata = query_data[i];
		const std::string& md5 = std::get<0>(qdata);
		const std::vector<float>& feats = std::get<1>(qdata);
		const std::vector<std::string>& gt_md5s = std::get<2>(qdata);
		std::priority_queue<std::pair<float, labeltype >> result = indobj.SearchKnn(feats.data(), top_n);
		std::vector<int> top_indexs;
		while(result.size())
		{
			top_indexs.emplace_back(result.top().second);
			result.pop();
		}
		// reverse
		std::reverse(top_indexs.begin(), top_indexs.end());
		// get md5
		std::vector<std::string> top_md5s;
		for(auto ind : top_indexs) top_md5s.emplace_back(labelmap[ind]);

		ofs << " - Query info[i, md5]: [ " << i << ", " << md5 << " ]" << std::endl;
		std::cout << " - Query info[i, md5]: [" << i << ", " << md5 << "]" << std::endl;
		for(int n = 0; n < std::min(10, (int)top_indexs.size()); ++n)
		{
			ofs << top_indexs[n] << ":" << top_md5s[n] << ", ";
		}
		ofs << std::endl;
		std::vector<float> recs = eval(top_md5s, gt_md5s, cursors);	
		for(int n = 0; n < recs.size(); ++n)
		{
			float val = recs[n]  + avg_recalls[n] * i;
			avg_recalls[n] = val / (i + 1);
			std::cout << " -This Recall@" << cursors[n] << ":\t" << recs[n] 
					  << ", Avg Recall@" << cursors[n] << ":\t" << avg_recalls[n] << std::endl;
			ofs << " -This Recall@" << cursors[n] << ":\t" << recs[n] 
					  << ", Avg Recall@" << cursors[n] << ":\t" << avg_recalls[n] << std::endl;
		}	
	}
	ofs.close();
	
	return 0;
}

std::vector<float> eval(const std::vector<std::string>& top_md5s, const std::vector<std::string>& gt_md5s, const std::vector<int>& cursors)
{
	std::vector<float> recs;
	if(gt_md5s.empty()) return recs;

	std::vector<int> index;
	for(auto& md5 : gt_md5s)
	{
		auto itr = std::find(top_md5s.begin(), top_md5s.end(), md5);
		int inds = std::distance(top_md5s.begin(), itr);
		index.emplace_back(inds);
	} 
	for(int i = 0; i < cursors.size(); ++i)
	{
		const int cursor = cursors[i];
		size_t hitnum = std::count_if(index.begin(), index.end(), [&cursor](int x)
		{
			return x < cursor;	
		});
		recs.emplace_back((float)hitnum / gt_md5s.size());
	}

	return recs;
}
inline void showProgressBar(int cnt, int total, int interval)
{   
    if(cnt != total && cnt % interval != 0) return;
    static auto pre = system_clock::now();
    auto now = system_clock::now();

    auto duration = duration_cast<microseconds>(now - pre);
    double cost = double(duration.count()) * microseconds::period::num / microseconds::period::den + 0.0000001;
    pre = now;
    int speed = interval / cost;

    float ratio = float(cnt) / total;
    string bar(ratio * 100, '#');
    const char *lable = "|/-\\";
    printf("[%-100s][%d%%, %d/%d, %d/s] [%c]\r", bar.c_str(), int(ratio * 100), cnt, total, speed, lable[cnt/interval%4]);    
    fflush(stdout);
    if(cnt == total) cout << endl;
}

int countLines(const char *filename)
{
    ifstream ReadFile;
    int n=0;
    string tmp;
    ReadFile.open(filename,ios::in);//ios::in 表示以只读的方式读取文件
    if(ReadFile.fail())//文件打开失败:返回0
    {
        return 0;
    }
    else//文件存在
    {
        while(getline(ReadFile,tmp,'\n'))
        {
            //cout << "tmp: " << tmp << endl;
            n += !tmp.empty();
        }
        ReadFile.close();
        return n;
    }
}


void loadTrainData(string& file, std::map<int, vector<float> >& db, std::map<int, std::string>& labelmap) 
{
	std::cout << " - loadTrainData : " << file << std::endl;
	const int dim = DIM;
    int totalLine = countLines(file.c_str());   
    char str[1000];
    FILE* pDBFile = fopen(file.c_str(), "r");
    for(int i = 0; i < totalLine; ++i)
    {   
        showProgressBar(i + 1, totalLine, 1000);
        fscanf(pDBFile, "%s", str);
        string md5 = str;
		std::vector<float> vec;
		for(int n = 0; n < dim; ++n)
		{
        	fscanf(pDBFile, "%s", str);
			vec.emplace_back(std::stof(str));
		}
        db[i] = std::move(vec);
		labelmap[i] = md5;
    }   
    fclose(pDBFile);
}

void loadQueryData(string& file, std::vector<QueryDataType>& db) 
{
	std::cout << " - loadQueryData : " << file << std::endl;
	const int dim = DIM;
    int totalLine = countLines(file.c_str());   
    char str[1000];
    FILE* pDBFile = fopen(file.c_str(), "r");
    for(int i = 0; i < totalLine; ++i)
    {   
		QueryDataType tmp;
        showProgressBar(i + 1, totalLine, 100);
        fscanf(pDBFile, "%s", str);
        string md5 = str;
		std::vector<float> vec;
		for(int n = 0; n < dim; ++n)
		{
        	fscanf(pDBFile, "%s", str);
			vec.emplace_back(std::stof(str));
		}
		char sLineWord[10000];
		fscanf(pDBFile, "%[^\n]%*c", sLineWord);
		char head[] = "\t";
		char *strip = NULL;
		strip = strtok( sLineWord, head );
		strip = strtok( NULL, head );
		std::vector<std::string> concat_md5s;
		char delims[] = " ";
		char* result = strtok( strip, delims );
		while( result != NULL ) 
		{
			concat_md5s.emplace_back(result);
			result = strtok( NULL, delims );
		}     	
		// Take Top-10
		if(concat_md5s.size() > 10) concat_md5s.resize(10);

		std::get<0>(tmp) = std::move(md5);	
		std::get<1>(tmp) = std::move(vec);	
		std::get<2>(tmp) = std::move(concat_md5s);	
		db.emplace_back(std::move(tmp));
		//cout << " - md5: " << std::get<0>(db.back()) << endl;
		//cout << " - feat.size: " << std::get<1>(db.back()).size() << endl;
		//cout << " - md5s.size: " << std::get<2>(db.back()).size() << "," << std::get<2>(db.back()).front() << endl;
		//if(i==3)exit(0);
    }   
    fclose(pDBFile);
}
