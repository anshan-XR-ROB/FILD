#include "lcdengine.hpp"
#include "hnswlib.hpp"
//#include "geometrical_verification.hpp"
#include "utils.hpp"
//
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <algorithm>
#include <vector>


LCDEngine::LCDEngine()
{

    OnInit();
}
void LCDEngine::OnInit()
{
    // - read frame names
    const std::string& root_dir = "your_path_to/loop-closure-detection/"; 
    const std::string& image_dir = "resource/newcollege_right_2624/"; 
    const std::string& namelist_file = "resource/newcollege_right_2624_image_list.txt";
    frame_abspath_vec_ = std::move(ReadFrameAbsPaths(root_dir + namelist_file, root_dir + image_dir)); 
    std::cout << " - read frame names done." << std::endl;

    // - initial hnsw engine
    hnsw_engine_.reset(new Index("cosine", 1024));
    hnsw_engine_->CreateNewIndex(frame_abspath_vec_.size(), 48, 20); //16,200
    std::cout << " - initial hnsw engine done." << std::endl;
 
}

std::vector<std::string> 
LCDEngine::ReadFrameAbsPaths(const std::string& namelist_file, const std::string& image_dir)
{
    std::vector<std::string> abs_paths;
    std::ifstream ifs(namelist_file);	

	if(ifs.is_open() == false)
	{
		std::cerr << " - ReadFrameAbsPaths cannot open " << namelist_file << std::endl;
		return abs_paths;
	}

	char buffer[1000];
	while (!ifs.eof())
	{
		ifs.getline(buffer, 1000);
		std::string line(buffer);
		if (line.empty()) continue;
        line = Utils::Trim(line);
        abs_paths.emplace_back(image_dir + "/" + line);
	}

	ifs.close();

    return abs_paths;

}

int getLine(const char* filename)
{
  char flag;
  int linenum = 0;
  int count = 0;
  FILE *fp = fopen(filename,"rt+");
  while(!feof(fp)){
    flag=fgetc(fp);
    if(flag=='\n') count++ ;
  }
  linenum = count;
  fclose(fp);
  return linenum;
} 

std::pair<std::vector<float>, std::vector<cv::KeyPoint> >
GetDelfDescriptorsAndKeyPoints(string &descname, string &pointname)
{    
    std::vector<float> desc;
    std::vector< cv::KeyPoint > keypoints;

    int line_des = getLine(descname.c_str());
    FILE *fp_des = fopen(descname.c_str(), "r");
    if(fp_des == NULL)
    {
        printf("File:%s load error!\n", descname.c_str());
        return std::make_pair(desc, keypoints);
    }
    int line_loc = getLine(pointname.c_str());
    FILE *fp_loc = fopen(pointname.c_str(), "r");
    if(fp_loc == NULL)
    {
        printf("File:%s load error!\n", pointname.c_str());
        return std::make_pair(desc, keypoints);
    }
    if(line_des != line_loc)
    {
        printf("The length of descriptors is not equal to the keypoint!\n");
        return std::make_pair(desc, keypoints);
    }

    int dim = 40;
	std::vector< cv::Point2f > points;
	int ind = 0;
	for(int i = 0; i < line_des; i++)
	{
		for(int j = 0; j < dim; j++)
		{
			float tmp = 0.0;
			fscanf(fp_des, "%f", &tmp);
			desc.push_back(tmp);
		}
		float x = 0.0, y = 0.0;
		fscanf(fp_loc, "%f %f", &x,&y);
		points.push_back(Point2f(y,x));
    }
	cv::KeyPoint::convert(points, keypoints);
	fclose(fp_des);
	fclose(fp_loc);

	return std::make_pair(std::move(desc), std::move(keypoints));
}

std::vector<float> loadGlobalFeature(string &gfeatname)
{
	std::vector<float> desc;

    int line_des = getLine(gfeatname.c_str());
    FILE *fp_des = fopen(gfeatname.c_str(), "r");
    if(fp_des == NULL)
    {
        printf("File:%s load error!\n", gfeatname.c_str());
        return desc;
    }

    int dim = 1024;
	int ind = 0;
	for(int j = 0; j < dim; j++)
	{
		float tmp = 0.0;
		fscanf(fp_des, "%f", &tmp);
		desc.push_back(tmp);
	}
	fclose(fp_des);

	return std::move(desc);
}

void LCDEngine::Run()
{
    const int search_avoid_frames = 1*40;
    const int top_n = 5;
    const float threshold = 0.1;          
    const int dim = 40;
    
    //string outname = "lcd_result_kitti05_glo05_50w_0.6binary_256_M48_EF40_top20_delf+hash_0323_lo20_white_54.txt"; 
    string outname = "lcd_result.txt"; 
    std::ofstream ofs(outname); 
    //string time_txt_name = "kitti00_BF_knnmatch_time.txt"; 
    //std::ofstream time_ofs(time_txt_name); 
    std::list<std::pair<int, std::vector<float> > > inds_buf;
    std::vector<std::vector<float> > descriptor_vec;
    std::vector<std::vector<cv::KeyPoint> > keypoint_vec;

    double extract_cnn_time = 0.0; 
    double extract_surf_time = 0.0;
    double create_hash_time = 0.0;
    double add_feature_time = 0.0;
    double search_knn_time = 0.0;
    double knn_match_time = 0.0;
    double ransac_time = 0.0;
    int perform_ransac = 0;
	int perform_knnmatch = 0;
    vector<double> time_vec;
    vector<double> time_vec_extract_cnn_time;
    vector<double> time_vec_extract_surf_time;
    // vector<double> time_vec_create_hash_time;
    vector<double> time_vec_add_feature_time;
    vector<double> time_vec_search_knn_time;
    vector<double> time_vec_hash_match_time;
    vector<double> time_vec_ransac_time;

    // if we save all surf
    //std::vector<std::vector<float> > surf_vec;
    // load each frame
    for(size_t i = 0; i != frame_abspath_vec_.size(); ++i)
    {
        float current_time = 0.0;
        std::cout << "start: " << i << std::endl;
        const std::string& path = frame_abspath_vec_[i]; 
        cv::Mat frame = cv::imread(path, 1); 
        //get imagename
        //string pngname = path.substr(path.length()-10, 6);   //kitti   StLucia
        //string pngname = path.substr(path.length()-33, 29);  //malaga
        //string pngname = path.substr(path.length()-27, 23);  //bicocca
        string pngname = path.substr(path.length()-40, 36);  //newcollege
        //string pngname = path.substr(path.length()-23, 19);   //mh05
        //string pngname = path.substr(path.length()-8, 4);     //city_centre
        string descpath = "your_path_to/loop-closure-detection/middata/feature/";
        string descname = descpath + pngname + "_des.txt";
        string pointname = descpath + pngname + "_loc.txt"; 
        auto start = system_clock::now();
        //std::vector<float> feat = std::move(extractor_->DoInference(frame));
        string gfeatpath = "your_path_to/loop-closure-detection/middata/feature/";
        string gfeatname = gfeatpath + pngname + "_global.txt";
        std::vector<float> feat = loadGlobalFeature(gfeatname);
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        double time = double(duration.count()) * microseconds::period::num / microseconds::period::den;     
        extract_cnn_time += time;
        current_time += time;
        time_vec_extract_cnn_time.push_back(time);
        std::cout << "RILD::Extract CNN feature cost " << time << " sec " << std::endl;         
        
        start = system_clock::now();
	std::pair<std::vector<float>, std::vector<cv::KeyPoint> > surf = GetDelfDescriptorsAndKeyPoints(descname, pointname);
        std::vector<float> descriptor = surf.first;
		descriptor_vec.push_back(surf.first);
        std::vector<cv::KeyPoint> keypoint = surf.second;
        keypoint_vec.push_back(keypoint);
        end = system_clock::now();
        duration = duration_cast<microseconds>(end - start);
        time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        extract_surf_time += time;
        current_time += time;
        time_vec_extract_surf_time.push_back(time);
        std::cout << "RILD::Extract SURF feature cost " << time << " sec " << std::endl;        

        if(i > search_avoid_frames)
        {
            auto guard_data = std::move(inds_buf.front());
            inds_buf.pop_front();
            int guard_ind                 = std::get<0>(guard_data);
            std::vector<float> guard_feat = std::get<1>(guard_data);
            // insert into hnsw_engine
            start = system_clock::now();
            hnsw_engine_->AddItem((void *)guard_feat.data(), guard_ind);
            end = system_clock::now();
            duration = duration_cast<microseconds>(end - start);
            time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            add_feature_time += time;
            current_time += time;
            time_vec_add_feature_time.push_back(time);
            std::cout << "RILD::Add feature into graph cost " << time << " sec " << std::endl;
            // do query use hnsw_engine
            start = system_clock::now();
            std::priority_queue<std::pair<float, hnswlib::labeltype> > result = 
                    hnsw_engine_->SearchKnn((void *)feat.data(), top_n);
            end = system_clock::now();
            duration = duration_cast<microseconds>(end - start);
            time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            search_knn_time += time;
            current_time += time;
            time_vec_search_knn_time.push_back(time);
            std::cout << "RILD::Search the knn cost " << time << " sec " << std::endl;            

            std::vector<std::pair<float, hnswlib::labeltype> > top_indexs;
            while(result.size())
            {
                top_indexs.emplace_back(result.top());
                result.pop();
            }
            // reverse indes 
            std::reverse(top_indexs.begin(), top_indexs.end());


            ofs << i + 1;
            // do geom verif 
            //start = system_clock::now();
            double knnmatch_time_of_frame = 0.0;
            for(int j = 0; j < top_indexs.size(); j++)
            {
               //if( top_indexs[j].first < (1.0 - threshold))
               {
                   //cout << "Match top_indexs[j].second: " << top_indexs[j].second << endl;
                   //cout << "hash_vec size: " << hash_vec.size() << endl;
                   start = system_clock::now();
                   FlannBasedMatcher matcher;
                   //BFMatcher matcher;
                   vector< vector< DMatch> > matches;
                   Mat descriptors_object = Mat(descriptor.size()/40, 40, CV_32FC1);
                   Mat descriptors_scene = Mat(descriptor_vec[top_indexs[j].second].size()/40, 40, CV_32FC1);
		   //printf("====%d,%d=====",descriptor.size(),descriptor_vec[top_indexs[j].second].size());
                   memcpy(descriptors_object.data, descriptor.data(), descriptor.size()*sizeof(float));
                   memcpy(descriptors_scene.data, descriptor_vec[top_indexs[j].second].data(), descriptor_vec[top_indexs[j].second].size()*sizeof(float));

				   matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2 );

                   std::vector< DMatch > good_matches;
                   for (int k = 0; k < std::min(int(descriptor_vec[top_indexs[j].second].size()/40) - 1, (int)matches.size()); k++)
                   {   
                       if ( (matches[k][0].distance < 0.7*(matches[k][1].distance)) &&
                               ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
                       {   
                           // take the first result only if its distance is smaller than 0.6*second_best_dist
                           // that means this descriptor is ignored if the second distance is bigger or of similar
                           good_matches.push_back( matches[k][0] );
                       }   
                   }   
				   end = system_clock::now();
				   duration = duration_cast<microseconds>(end - start);
				   time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
				   current_time += time;
				   knn_match_time += time;
				   knnmatch_time_of_frame += time;
				   perform_knnmatch++;
				   std::cout << "Matcher::knnMatch cost " << time << " sec " << std::endl;
                   
				   
				   start = system_clock::now();
				   int ransacPoints = 0;
                   if(good_matches.size() > 4)// 15)
                   {   
                       perform_ransac++;
					   std::vector<cv::Point2f> points1;
                       std::vector<cv::Point2f> points2;
                       for(int n = 0; n < good_matches.size(); n++)
                       {   
                           points1.push_back(keypoint_vec[i].at(good_matches[n].queryIdx).pt);
                           points2.push_back(keypoint_vec[top_indexs[j].second].at(good_matches[n].trainIdx).pt);
                       }   
                       std::vector<uchar> inliers(points1.size(),0);
                       cv::Mat fundamental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2),inliers,CV_FM_RANSAC);
					   //cv::Mat Homography= cv::findHomography(cv::Mat(points1),cv::Mat(points2), inliers,CV_RANSAC,5);  
                       for(int n = 0; n < inliers.size(); n++)
                       {   
                           if((unsigned int)inliers[n])
                           {   
                               ransacPoints++;
                           }   
                       }
                   }
                   end = system_clock::now();
                   duration = duration_cast<microseconds>(end - start);
                   time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                   ransac_time += time;
                   current_time += time;
                   time_vec_ransac_time.push_back(time);
                   std::cout << "RILD::RANSAC cost " << time << " sec " << std::endl;
                   // cout << "Process: " << i << " with surf matches:" << result.size() << "and inliers: " << ransacPoints << std::endl; 
                   // if(ransacPoints >= 7)
                   {
                      ofs << " " << top_indexs[j].second + 1 << " " << 1.0 - top_indexs[j].first << " " << good_matches.size() << " " << ransacPoints;
                   }
				   
               }
            }
            //time_ofs << knnmatch_time_of_frame*1000 <<std::endl; 
            //ofs << std::endl;
        }
        inds_buf.emplace_back(i, feat);
        time_vec.push_back(current_time);
    }

    std::sort(time_vec.begin(), time_vec.end());
    std::cout << "Min processing_time: " << time_vec.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max processing_time: " << time_vec.back() * 1000 <<" ms" << std::endl;
    double sum = 0.;
    for_each(time_vec.begin(), time_vec.end(), [&sum](float x) { sum += x; });
    sum /= std::max(1, (int)time_vec.size());
    std::cout << "Mean processing_time: " << sum * 1000 <<" ms" << std::endl;
    double stdev_time = Utils::CalcStd(time_vec);
    std::cout << "Std processing_time: " << stdev_time * 1000 <<" ms" << std::endl;

    double mean_extract_cnn_time = extract_cnn_time / frame_abspath_vec_.size(); 
    double mean_extract_surf_time = extract_surf_time / frame_abspath_vec_.size();
    // double mean_create_hash_time = create_hash_time / frame_abspath_vec_.size();
    double mean_add_feature_time = add_feature_time / (frame_abspath_vec_.size() - search_avoid_frames);
    double mean_search_knn_time = search_knn_time / (frame_abspath_vec_.size() - search_avoid_frames);
    double mean_knn_match_time = knn_match_time / perform_knnmatch;
    double mean_ransac_time = ransac_time / perform_ransac;
	double geometrical_verification_time = (knn_match_time + ransac_time) / (frame_abspath_vec_.size() - search_avoid_frames);


    double stdev_extract_cnn_time = Utils::CalcStd(time_vec_extract_cnn_time); 
    double stdev_extract_surf_time = Utils::CalcStd(time_vec_extract_surf_time);
    // double stdev_create_hash_time = Utils::CalcStd(time_vec_create_hash_time);
    double stdev_add_feature_time = Utils::CalcStd(time_vec_add_feature_time);
    double stdev_search_knn_time = Utils::CalcStd(time_vec_search_knn_time);
    //double stdev_hash_match_time = Utils::CalcStd(time_vec_hash_match_time);
    double stdev_ransac_time = Utils::CalcStd(time_vec_ransac_time);

    std::sort(time_vec_extract_cnn_time.begin(), time_vec_extract_cnn_time.end());
    std::cout << "Min extract_cnn_time: " << time_vec_extract_cnn_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max extract_cnn_time: " << time_vec_extract_cnn_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std extract_cnn_time: " << stdev_extract_cnn_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_extract_surf_time.begin(), time_vec_extract_surf_time.end());
    std::cout << "Min extract_surf_time: " << time_vec_extract_surf_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max extract_surf_time: " << time_vec_extract_surf_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std extract_surf_time: " << stdev_extract_surf_time * 1000 <<" ms" << std::endl;

    // std::sort(time_vec_create_hash_time.begin(), time_vec_create_hash_time.end());
    // std::cout << "Min create_hash_time: " << time_vec_create_hash_time.front() * 1000 <<" ms" << std::endl;
    // std::cout << "Max create_hash_time: " << time_vec_create_hash_time.back() * 1000 <<" ms" << std::endl;
    // std::cout << "Std create_hash_time: " << stdev_create_hash_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_add_feature_time.begin(), time_vec_add_feature_time.end());
    std::cout << "Min add_feature_time: " << time_vec_add_feature_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max add_feature_time: " << time_vec_add_feature_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std add_feature_time: " << stdev_add_feature_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_search_knn_time.begin(), time_vec_search_knn_time.end());
    std::cout << "Min search_knn_time: " << time_vec_search_knn_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max search_knn_time: " << time_vec_search_knn_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std search_knn_time: " << stdev_search_knn_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_ransac_time.begin(), time_vec_ransac_time.end());
    std::cout << "Min ransac_time: " << time_vec_ransac_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max ransac_time: " << time_vec_ransac_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std ransac_time: " << stdev_ransac_time * 1000 <<" ms" << std::endl;

    //mean time
    std::cout << "mean_extract_cnn_time: " << mean_extract_cnn_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_extract_surf_time: " << mean_extract_surf_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_add_feature_time: " << mean_add_feature_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_search_knn_time: " << mean_search_knn_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_knn_match_time: " << mean_knn_match_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_ransac_time: " << mean_ransac_time * 1000 <<" ms" << std::endl;
	std::cout << "The geometrical verification time: " << geometrical_verification_time * 1000 << " ms" << std::endl;
    Utils::printMemoryUsage();
    ofs.close();
	time_ofs.close();
}

void LCDEngine::OnDestroy()
{

}

