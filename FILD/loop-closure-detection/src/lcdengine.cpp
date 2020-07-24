#include "lcdengine.hpp"
#include "hnswlib.hpp"
#include "extractor.hpp"
#include "geometrical_verification.hpp"
#include "theialib.hpp"

LCDEngine::LCDEngine()
{

    OnInit();
}
void LCDEngine::OnInit()
{
    // - read frame names
    const std::string& root_dir = Utils::GetParamsOf<std::string>("root_dir"); 
    const std::string& image_dir = Utils::GetParamsOf<std::string>("image_dir"); 
    const std::string& namelist_file = Utils::GetParamsOf<std::string>("image_name_list");
    frame_abspath_vec_ = std::move(ReadFrameAbsPaths(root_dir + namelist_file, root_dir + image_dir)); 
    std::cout << " - read frame names done." << std::endl;

    // - initial hnsw engine
    hnsw_engine_.reset(new Index("cosine", Utils::GetParamsOf<int>("cnn_feat_dim")));
    hnsw_engine_->CreateNewIndex(frame_abspath_vec_.size(), Utils::GetParamsOf<int>("hnsw_m"), Utils::GetParamsOf<int>("hnsw_ef")); //16,200
    std::cout << " - initial hnsw engine done." << std::endl;

    // - initial cnn extratcor
    const std::string deploy = root_dir + "/resource/model_17000/no_bn.prototxt";
    const std::string weight = root_dir + "/resource/model_17000/no_bn.caffemodel";
    //const std::string deploy = root_dir + "/resource/model_27w/mobilenetv2_deploy0225.prototxt"; //"/resource/model_res152/deploy_resnet152_places365.prototxt";
    //const std::string weight = root_dir + "/resource/model_27w/mobilev2_4_iter_270000.caffemodel";//"/resource/model_res152/resnet152_places365.caffemodel";
    const int cnn_gpu_id = 3;
    extractor_.reset(new Extractor(deploy, weight, cnn_gpu_id));
    std::cout << " - initial cnn extractor done." << std::endl;
 
    // - initial geom verif
    const int surf_gpu_id = 2;
    const bool extended = true;
    geom_verif_.reset(new GeometricalVerification(surf_gpu_id, extended));
    std::cout << " - initial geom verif done." << std::endl;
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

void LCDEngine::Run()
{
    const int search_avoid_frames = Utils::GetParamsOf<int>("frame_rate") *
            Utils::GetParamsOf<int>("avoid_search_time");
    const int top_n = Utils::GetParamsOf<int>("top_n");
    const float threshold = Utils::GetParamsOf<float>("similarity_threshold");          
    const int dim = 128;
    
    string outname = "lcd_result_newcollege_17000_0.7_256_M48_EF40_top1.txt"; 
    std::ofstream ofs(outname); 
    std::list<std::pair<int, std::vector<float> > > inds_buf;
    std::vector<std::shared_ptr<HashedImage> > hash_vec;
    std::vector<std::vector<float> > descriptor_vec;
    std::vector<std::vector<cv::KeyPoint> > keypoint_vec;
    TheiaTool theia(dim);   

    double extract_cnn_time = 0.0; 
    double extract_surf_time = 0.0;
    double create_hash_time = 0.0;
    double add_feature_time = 0.0;
    double search_knn_time = 0.0;
    double hash_match_time = 0.0;
    double ransac_time = 0.0;
    int perform_ransac = 0;
    vector<double> time_vec;
    vector<double> time_vec_extract_cnn_time;
    vector<double> time_vec_extract_surf_time;
    vector<double> time_vec_create_hash_time;
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
          
        auto start = system_clock::now();
        std::vector<float> feat = std::move(extractor_->DoInference(frame));
        auto end = system_clock::now();
        auto duration = duration_cast<microseconds>(end - start);
        double time = double(duration.count()) * microseconds::period::num / microseconds::period::den;     
        extract_cnn_time += time;
        current_time += time;
        time_vec_extract_cnn_time.push_back(time);
        std::cout << "RILD::Extract CNN feature cost " << time << " sec " << std::endl;         
        
        start = system_clock::now();
        std::pair<std::vector<float>, std::vector<cv::KeyPoint> > surf = geom_verif_->GetSurfDescriptorsAndKeyPoints(frame);   
        //std::vector<float> descriptor = surf.first;
        //descriptor_vec.push_back(descriptor);
        //surf_vec.push_back(surf.first);
        std::vector<cv::KeyPoint> keypoint = surf.second;
        keypoint_vec.push_back(keypoint);
        end = system_clock::now();
        duration = duration_cast<microseconds>(end - start);
        time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        extract_surf_time += time;
        current_time += time;
        time_vec_extract_surf_time.push_back(time);
        std::cout << "RILD::Extract SURF feature cost " << time << " sec " << std::endl;        

        start = system_clock::now(); 
        std::shared_ptr<HashedImage> hash = theia.CreateHashedDescriptors(surf.first);               
        hash_vec.push_back(hash); 
        end = system_clock::now();
        duration = duration_cast<microseconds>(end - start);
        time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
        create_hash_time += time;
        current_time += time;
        time_vec_create_hash_time.push_back(time);
        std::cout << "RILD::Create Hash codes cost " << time << " sec " << std::endl;
        //std::cout << "hash ok " << "in frame " << i << std::endl;
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

            // sort top_indexs by euclidean distance, seems useless for our system
            /*std::multimap<double, int> sortmmp;
            std::vector<int> sortinds(top_indexs.size());
            for(int j = 0; j < top_indexs.size(); j++)
            {
                std::vector<float> point = std::move(hnsw_engine_->GetPoint(top_indexs[j].second)); 
                double eucldis = Utils::EuclDist(feat, point); 
                sortmmp.insert({eucldis, top_indexs[j].second});
            }
            // print
            std::cout << " - raw :" << std::endl;
            for(int j = 0; j < top_indexs.size(); j++)
            {
                std::cout << top_indexs[j].second << ", " << std::endl;
            }
        
            std::cout << "\n - new :" << std::endl;
            for(auto& pr : sortmmp)
            {
                std::cout << pr.second << ", " << std::endl;
            }
            std::cout << std::endl;*/

            ofs << i + 1;
            // do geom verif 
            //start = system_clock::now();
            for(int j = 0; j < top_indexs.size(); j++)
            {
               //if( top_indexs[j].first < (1.0 - threshold))
               {
                   //cout << "Match top_indexs[j].second: " << top_indexs[j].second << endl;
                   //cout << "hash_vec size: " << hash_vec.size() << endl;
                   start = system_clock::now();
                   std::vector<IndexedFeatureMatch> result = theia.Match(hash, hash_vec[top_indexs[j].second]); 
                   //std::vector<IndexedFeatureMatch> result = theia.Match(hash, descriptor, hash_vec[top_indexs[j].second], descriptor_vec[top_indexs[j].second]);
                   end = system_clock::now();
                   duration = duration_cast<microseconds>(end - start);
                   time = double(duration.count()) * microseconds::period::num / microseconds::period::den;
                   hash_match_time += time;
                   current_time += time;
                   time_vec_hash_match_time.push_back(time);
                   std::cout << "RILD::Matching using hash codes cost " << time << " sec " << std::endl;   
                    
                   start = system_clock::now();
                   int ransacPoints = 0;
                   if(result.size() > 15)
                   {   
                       perform_ransac++;
                       std::vector<cv::Point2f> points1;
                       std::vector<cv::Point2f> points2;
                       for(int n = 0; n < result.size(); n++)
                       {   
                           points1.push_back(keypoint_vec[i].at(result[n].feature1_ind).pt);
                           points2.push_back(keypoint_vec[top_indexs[j].second].at(result[n].feature2_ind).pt);
                       }   
                       std::vector<uchar> inliers(points1.size(),0);
                       cv::Mat fundemental= cv::findFundamentalMat(cv::Mat(points1),cv::Mat(points2),inliers,CV_FM_RANSAC);
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
                   //cout << "Process: " << i << " with surf matches:" << result.size() << "and inliers: " << ransacPoints << std::endl; 
                   //if(ransacPoints >= 7)
                   {
                      ofs << " " << top_indexs[j].second + 1 << " " << 1.0 - top_indexs[j].first << " " << result.size() << " " << ransacPoints;
                   }
               }
            }
            ofs << std::endl;
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
    double mean_create_hash_time = create_hash_time / frame_abspath_vec_.size();
    double mean_add_feature_time = add_feature_time / (frame_abspath_vec_.size() - search_avoid_frames);
    double mean_search_knn_time = search_knn_time / (frame_abspath_vec_.size() - search_avoid_frames);
    double mean_hash_match_time = hash_match_time / (frame_abspath_vec_.size() - search_avoid_frames);
    double mean_ransac_time = ransac_time / perform_ransac;


    double stdev_extract_cnn_time = Utils::CalcStd(time_vec_extract_cnn_time); 
    double stdev_extract_surf_time = Utils::CalcStd(time_vec_extract_surf_time);
    double stdev_create_hash_time = Utils::CalcStd(time_vec_create_hash_time);
    double stdev_add_feature_time = Utils::CalcStd(time_vec_add_feature_time);
    double stdev_search_knn_time = Utils::CalcStd(time_vec_search_knn_time);
    double stdev_hash_match_time = Utils::CalcStd(time_vec_hash_match_time);
    double stdev_ransac_time = Utils::CalcStd(time_vec_ransac_time);

    std::sort(time_vec_extract_cnn_time.begin(), time_vec_extract_cnn_time.end());
    std::cout << "Min extract_cnn_time: " << time_vec_extract_cnn_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max extract_cnn_time: " << time_vec_extract_cnn_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std extract_cnn_time: " << stdev_extract_cnn_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_extract_surf_time.begin(), time_vec_extract_surf_time.end());
    std::cout << "Min extract_surf_time: " << time_vec_extract_surf_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max extract_surf_time: " << time_vec_extract_surf_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std extract_surf_time: " << stdev_extract_surf_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_create_hash_time.begin(), time_vec_create_hash_time.end());
    std::cout << "Min create_hash_time: " << time_vec_create_hash_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max create_hash_time: " << time_vec_create_hash_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std create_hash_time: " << stdev_create_hash_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_add_feature_time.begin(), time_vec_add_feature_time.end());
    std::cout << "Min add_feature_time: " << time_vec_add_feature_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max add_feature_time: " << time_vec_add_feature_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std add_feature_time: " << stdev_add_feature_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_search_knn_time.begin(), time_vec_search_knn_time.end());
    std::cout << "Min search_knn_time: " << time_vec_search_knn_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max search_knn_time: " << time_vec_search_knn_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std search_knn_time: " << stdev_search_knn_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_hash_match_time.begin(), time_vec_hash_match_time.end());
    std::cout << "Min hash_match_time: " << time_vec_hash_match_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max hash_match_time: " << time_vec_hash_match_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std hash_match_time: " << stdev_hash_match_time * 1000 <<" ms" << std::endl;

    std::sort(time_vec_ransac_time.begin(), time_vec_ransac_time.end());
    std::cout << "Min ransac_time: " << time_vec_ransac_time.front() * 1000 <<" ms" << std::endl;
    std::cout << "Max ransac_time: " << time_vec_ransac_time.back() * 1000 <<" ms" << std::endl;
    std::cout << "Std ransac_time: " << stdev_ransac_time * 1000 <<" ms" << std::endl;
    //mean time
    std::cout << "mean_extract_cnn_time: " << mean_extract_cnn_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_extract_surf_time: " << mean_extract_surf_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_create_hash_time: " << mean_create_hash_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_add_feature_time: " << mean_add_feature_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_search_knn_time: " << mean_search_knn_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_hash_match_time: " << mean_hash_match_time * 1000 <<" ms" << std::endl;
    std::cout << "mean_ransac_time: " << mean_ransac_time * 1000 <<" ms" << std::endl;
    Utils::printMemoryUsage();
    ofs.close();
}

void LCDEngine::OnDestroy()
{

}

