#include "hnswlib.hpp"

Index::Index(const std::string &space_name, const int dim)
	:dim_(dim)
{
	normalize_ = false;

	if(space_name=="l2") {
		dist_space_.reset(new hnswlib::L2Space(dim));
	}
	else if(space_name=="ip") {
		dist_space_.reset(new hnswlib::InnerProductSpace(dim));
	}
	else if(space_name=="cosine") {
		dist_space_.reset(new hnswlib::InnerProductSpace(dim));
		normalize_ = true;
	}

}

void Index::CreateNewIndex(const size_t maxElements, const size_t M, const size_t efConstruction, const size_t random_seed)
{
	appr_alg_.reset(new hnswlib::HierarchicalNSW<float>(dist_space_.get(), maxElements, M, efConstruction, random_seed));	
}

void Index::Set_ef(size_t ef)
{
	appr_alg_->ef_ = ef;
}

void Index::SaveIndex(const std::string &path_to_index)
{
	appr_alg_->saveIndex(path_to_index);
}

void Index::LoadIndex(const std::string &path_to_index, size_t max_elements)
{
	appr_alg_.reset(new hnswlib::HierarchicalNSW<float>(dist_space_.get(), path_to_index, false, max_elements));
}

void Index::AddItem(void *datapoint, hnswlib::labeltype label)
{
	if(normalize_ == true)
		NormalizeVector((float*)datapoint, (float*)datapoint);

	appr_alg_->addPoint(datapoint, label);	
}

std::vector<float> Index::GetPoint(hnswlib::labeltype label)
{
    return std::move(appr_alg_->getPointByLabel(label));
}

std::priority_queue<std::pair<float, hnswlib::labeltype > > 
Index::SearchKnn(const void *query_data, size_t k)
{
	if(normalize_ == true)
		NormalizeVector((float*)query_data, (float*)query_data);
	return appr_alg_->searchKnn(query_data, k);	
}

void Index::NormalizeVector(float *data, float *norm_array)
{
   float norm = 0.0f;
   for(int i = 0; i < dim_; i++)
       norm += data[i]*data[i];
   norm = 1.0f / (sqrtf(norm) + 1e-30f);
   for(int i = 0; i < dim_; i++)
       norm_array[i] = data[i] * norm;
}
