#pragma once

#include <queue>
#include <random>
#include <unordered_map>
#include <memory>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <atomic>
#include <unordered_set>
#include <unordered_map>
#include <fstream>
#include <deque>
#include <mutex>
#include <string.h>

#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>

#define  __builtin_popcount(t) __popcnt(t)
#else

#include <x86intrin.h>

#endif


#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif


namespace hnswlib {
    typedef unsigned short int vl_type;

    class VisitedList {
    public:
        vl_type curV;
        vl_type *mass;
        unsigned int numelements;

        VisitedList(int numelements1);

        void reset(); 

        ~VisitedList(); 
    };
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

    class VisitedListPool {
        std::deque<VisitedList *> pool;
        std::mutex poolguard;
        int numelements;

    public:
        VisitedListPool(int initmaxpools, int numelements1); 

        VisitedList *getFreeVisitedList();

        void releaseVisitedList(VisitedList *vl); 

        ~VisitedListPool(); 
    };

    typedef size_t labeltype;

    template<typename T>
    static void writeBinaryPOD(std::ostream &out, const T &podRef) {
        out.write((char *) &podRef, sizeof(T));
    }

    template<typename T>
    static void readBinaryPOD(std::istream &in, T &podRef) {
        in.read((char *) &podRef, sizeof(T));
    }

    template<typename MTYPE>
    using DISTFUNC = MTYPE(*)(const void *, const void *, const void *);


    template<typename MTYPE>
    class SpaceInterface {
    public:
        //virtual void search(void *);
        virtual size_t get_data_size() = 0;

        virtual DISTFUNC<MTYPE> get_dist_func() = 0;

        virtual void *get_dist_func_param() = 0;

    };

    template<typename dist_t>
    class AlgorithmInterface {
    public:
        virtual void addPoint(void *datapoint, labeltype label)=0;
        virtual std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *, size_t) const = 0;
        virtual void saveIndex(const std::string &location)=0;
        virtual ~AlgorithmInterface(){
        }
    };


    typedef unsigned int tableint;
    typedef unsigned int linklistsizeint;
	
	template<typename dist_t>
    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const &a,
                                  std::pair<dist_t, tableint> const &b) const noexcept {
            return a.first < b.first;
        }
    };

    template<typename dist_t>
    class HierarchicalNSW : public AlgorithmInterface<dist_t> {
    public:

        HierarchicalNSW(SpaceInterface<dist_t> *s); 
        HierarchicalNSW(SpaceInterface<dist_t> *s, const std::string &location, bool nmslib = false, size_t max_elements=0);
        HierarchicalNSW(SpaceInterface<dist_t> *s, size_t max_elements, size_t M = 16, size_t ef_construction = 200, size_t random_seed = 100);
        ~HierarchicalNSW();

        size_t max_elements_;
        size_t cur_element_count;
        size_t size_data_per_element_;
        size_t size_links_per_element_;

        size_t M_;
        size_t maxM_;
        size_t maxM0_;
        size_t ef_construction_;

        double mult_, revSize_;
        int maxlevel_;


        VisitedListPool *visited_list_pool_;
        std::mutex cur_element_count_guard_;

        std::vector<std::mutex> link_list_locks_;
        tableint enterpoint_node_;



        size_t size_links_level0_;
        size_t offsetData_, offsetLevel0_;


        char *data_level0_memory_;
        char **linkLists_;
        std::vector<int> element_levels_;


        size_t data_size_;
        size_t label_offset_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;
        std::unordered_map<labeltype, tableint> label_lookup_;

        std::default_random_engine level_generator_;

        inline labeltype getExternalLabel(tableint internal_id) const; 

        inline labeltype *getExternalLabeLp(tableint internal_id) const; 

        inline char *getDataByInternalId(tableint internal_id) const;

        int getRandomLevel(double reverse_size);

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst<dist_t> >
        searchBaseLayer(tableint enterpoint_id, void *data_point, int layer);
            
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst<dist_t> >
        searchBaseLayerST(tableint ep_id, const void *data_point, size_t ef) const; 

        void getNeighborsBySimple(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst<dist_t> > &top_candidates,
                const size_t M); 

        void getNeighborsByHeuristic2(
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst<dist_t> > &top_candidates,
                const size_t M); 


        linklistsizeint *get_linklist0(tableint internal_id);

        linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_);

        linklistsizeint *get_linklist(tableint internal_id, int level); 

        void mutuallyConnectNewElement(
            void *data_point, tableint cur_c,
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst<dist_t> > top_candidates,
            int level); 
        std::mutex global;
        size_t ef_;

        void setEf(size_t ef); 


        std::priority_queue<std::pair<dist_t, tableint>> searchKnnInternal(void *query_data, int k); 

        void saveIndex(const std::string &location); 
        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i=0); 

        template<typename data_t>
        std::vector<data_t> getDataByLabel(labeltype label)
		{
          tableint label_c = label_lookup_[label];
          char* data_ptrv = getDataByInternalId(label_c);
          size_t dim = *((size_t *) dist_func_param_);
          std::vector<data_t> data;
          data_t* data_ptr = (data_t*) data_ptrv;
          for (int i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
          }
          return data;
        };


        void addPoint(void *data_point, labeltype label);
        tableint addPoint(void *data_point, labeltype label, int level);
        std::vector<float> getPointByLabel(labeltype label);
        std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *query_data, size_t k) const; 

    };


    template<typename dist_t> class AlgorithmInterface;
    template<typename dist_t> class SpaceInterface;

    template<typename dist_t>
    class BruteforceSearch : public AlgorithmInterface<dist_t> {
    public:
        BruteforceSearch(SpaceInterface<dist_t> *s);
        BruteforceSearch(SpaceInterface<dist_t> *s, const std::string &location); 

        BruteforceSearch(SpaceInterface<dist_t> *s, size_t maxElements); 
        ~BruteforceSearch(); 
        void addPoint(void *datapoint, labeltype label);
        void removePoint(labeltype cur_external);
        std::priority_queue<std::pair<dist_t, labeltype >> searchKnn(const void *query_data, size_t k) const;
        void saveIndex(const std::string &location);
        void loadIndex(const std::string &location, SpaceInterface<dist_t> *s);

        char *data_;
        size_t maxelements_;
        size_t cur_element_count;
        size_t size_per_element_;

        size_t data_size_;
        DISTFUNC<dist_t> fstdistfunc_;
        void *dist_func_param_;

        std::unordered_map<labeltype,size_t > dict_external_to_internal;

    };

    static float
    InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr);

    static float
    InnerProductSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr);

    static float
    InnerProductSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr); 

    class InnerProductSpace : public SpaceInterface<float> {

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        InnerProductSpace(size_t dim); 

        size_t get_data_size();

        DISTFUNC<float> get_dist_func();

        void *get_dist_func_param();

    };


    static float
    L2Sqr(const void *pVect1, const void *pVect2, const void *qty_ptr);

    static float
    L2SqrSIMD16Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr);


    static float
    L2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr);

    class L2Space : public SpaceInterface<float>{

        DISTFUNC<float> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2Space(size_t dim);

        size_t get_data_size();

        DISTFUNC<float> get_dist_func();

        void *get_dist_func_param(); 

    };

    static int
    L2SqrI(const void *__restrict pVect1, const void *__restrict pVect2, const void *__restrict qty_ptr);

    class L2SpaceI : public SpaceInterface<int> {

        DISTFUNC<int> fstdistfunc_;
        size_t data_size_;
        size_t dim_;
    public:
        L2SpaceI(size_t dim);

        size_t get_data_size();

        DISTFUNC<int> get_dist_func();

        void *get_dist_func_param(); 

    };

}

class Index
{
public:
	Index(const std::string &space_name, const int dim);

	void CreateNewIndex(const size_t maxElements, const size_t M = 16, const size_t efConstruction = 200, const size_t random_seed = 100);

	void Set_ef(size_t ef);

	void SaveIndex(const std::string &path_to_index);

	void LoadIndex(const std::string &path_to_index, size_t max_elements);

	void AddItem(void *datapoint, hnswlib::labeltype label);

    std::vector<float> GetPoint(hnswlib::labeltype label);  

	std::priority_queue<std::pair<float, hnswlib::labeltype > > SearchKnn(const void *, size_t);

	void NormalizeVector(float *data, float *norm_array);

private:
	std::shared_ptr<hnswlib::HierarchicalNSW<float> > appr_alg_;	

	std::shared_ptr<hnswlib::SpaceInterface<float> > dist_space_;

	bool normalize_; 

	const int dim_;
};
