
#ifndef SPBAU_RECOMMENDER_COLLABORATIVE_FILTERING_HPP_
#define SPBAU_RECOMMENDER_COLLABORATIVE_FILTERING_HPP_

#include <string>

#include "grouplens.hpp"
#include "knn.hpp"

template <class D, class M, class S, class A>
class collaborative_filtering_algorithm_t
{
public:
	typedef D dataset_type;
	typedef M dataset_mask_type;
	typedef S similarity_type;
	typedef A average_type;

	collaborative_filtering_algorithm_t(const std::string &name)
		:m_name(name)
	{}

	std::string name() const
	{
		return m_name;
	}

	virtual void operator()(D &algo_prediction, const D &data, const M &mask, 
							S &users_similarity, 
							const A &avg_users_rating, 
							const A &avg_product_ratings) = 0;
private:
	const std::string m_name;
};

template <class D, class M, class S, class A>
class grouplens_algo_t : public collaborative_filtering_algorithm_t<D, M, S, A>
{
public:
	grouplens_algo_t()
		:collaborative_filtering_algorithm_t("GroupLens")
	{}
	virtual void operator()(D &algo_prediction, const D &data, const M &mask, 
							S &users_similarity, 
							const A &avg_users_rating, 
							const A &avg_products_rating)
	{
		grouplens(algo_prediction, 
				  data, mask, 
				  users_similarity, 
				  avg_users_rating, avg_products_rating);
	}
};

template <class D, class M, class S, class A>
class knn_grouplens_algo_t : public collaborative_filtering_algorithm_t<D, M, S, A>
{
public:
	knn_grouplens_algo_t()
		:collaborative_filtering_algorithm_t("k-NN-GroupLens")
	{}
	virtual void operator()(D &algo_prediction, const D &data, const M &mask, 
							S &users_similarity, 
							const A &avg_users_rating, 
							const A &avg_products_rating)
	{
		knn(algo_prediction, 2, 
			data, mask, 
			users_similarity, 
			avg_users_rating, avg_products_rating, 
			0);
	}
};

#endif	// SPBAU_RECOMMENDER_COLLABORATIVE_FILTERING_HPP_
