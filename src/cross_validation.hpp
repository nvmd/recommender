
#ifndef SPBAU_RECOMMENDER_CROSS_VALIDATION_HPP_
#define SPBAU_RECOMMENDER_CROSS_VALIDATION_HPP_

#include <iostream>
#include <string>
#include <memory>

#include <itpp/itbase.h>
#include <itpp/base/vec.h>
#include <itpp/base/mat.h>

#include "dataset_io.hpp"
#include "user_resemblance.hpp"
#include "error.hpp"
#include "collaborative_filtering.hpp"


template <class T>
void cross_validation_get_sets(const T &triplets, T &validation, T &learning, float validation_rel_size = 0.1)
{
	size_t triplets_count = triplets.size();
	size_t validation_size = triplets_count * validation_rel_size + 1;
	size_t learning_size = triplets_count - validation_size;

	validation.reserve(validation_size);
	learning.reserve(learning_size);

	size_t i = 0;
	while (i<validation_size)
	{
		validation.push_back(triplets[i]);
		++i;
	}
	while (i<triplets_count)
	{
		learning.push_back(triplets[i]);
		++i;
	}
}

template <class M, class V, class B>
void avg_ratings(const M &users_ratings, const B &users_ratings_mask, 
					V &avg_users_rating, V &avg_product_ratings)
{
	// Average user's ratings and product's ratings
#if defined(ALG_REF_IMPL) || !defined(ALG_REF_IMPL)
	for (int i=0; i<users_ratings.rows(); ++i)
	{
		size_t valuable = 0;
		for (int j=0; j<users_ratings.cols(); ++j)
		{
			if (bool(users_ratings_mask(i,j)) == true)
			{
				++valuable;
				avg_users_rating[i] += users_ratings(i,j);
			}
		}
		avg_users_rating[i] = valuable > 0 ? avg_users_rating[i]/valuable : 0;
	}
	
	for (int i=0; i<users_ratings.cols(); ++i)
	{
		size_t valuable = 0;
		for (int j=0; j<users_ratings.rows(); ++j)
		{
			if (bool(users_ratings_mask(j,i)) == true)
			{
				++valuable;
				avg_product_ratings[i] += users_ratings(j,i);
			}
		}
		avg_product_ratings[i] = valuable > 0 ? avg_product_ratings[i]/valuable : 0;
	}
#else
#error ALG_ITPP_IMPL implementation of avg_ratings is obsolete
	for (int i=0; i<users_ratings.rows(); ++i)
	{
		avg_users_rating[i] = itpp::sum(users_ratings.get_row(i));
	}
	for (int j=0; j<users_ratings.cols(); ++j)
	{
		avg_product_ratings[j] = itpp::sum(users_ratings.get_col(j));
	}
	avg_users_rating /= users_ratings.cols();
	avg_product_ratings /= users_ratings.rows();
#endif
}

template <class D, class M, typename AlgoInputIterator>
void validate_algorithms(const D &learning, const M &learning_mask,
						   const D &validation, const M &validation_mask,
						   AlgoInputIterator algo_begin, AlgoInputIterator algo_end,
						   size_t verbosity = 0)
{
	// Average user's and product's ratings
	if (verbosity >= 1)
	{
		std::cout << "Average user's and product's ratings...";
	}
	itpp::vec avg_users_rating(learning.rows());
	itpp::vec avg_products_rating(learning.cols());
	avg_users_rating.zeros();
	avg_products_rating.zeros();
	avg_ratings(learning, learning_mask, avg_users_rating, avg_products_rating);
	if (verbosity >= 1)
	{
		std::cout << "Done." << std::endl;
	}
	
	if (verbosity >= 2)
	{
		std::cout << "Avg. users' ratings: \n" << avg_users_rating << std::endl;
		std::cout << "Avg. products' ratings: \n" << avg_products_rating << std::endl;
	}

	// Users' resemblance
	itpp::mat user_resemblance(learning.rows(), learning.rows());
	itpp::bmat user_resemblance_mask(user_resemblance.rows(), user_resemblance.cols());
	user_resemblance.zeros();
	user_resemblance_mask.zeros();

	// compute users' resemblance on demand
	user_resemblance_itpp_t u_resemblance(learning, 
										  user_resemblance, user_resemblance_mask, 
										  correlation_coeff_resembl_metric_t());

	// Validate algorithms
	itpp::mat algo_prediction(learning.rows(), learning.cols());
	for (AlgoInputIterator i = algo_begin; i != algo_end; ++i)
	{
		algo_prediction.zeros();
		if (verbosity >= 1)
		{
			std::cout << (*i)->name();
		}
		(**i)(algo_prediction, learning, learning_mask, 
			  u_resemblance, avg_users_rating, avg_products_rating);
		if (verbosity >= 1)
		{
			std::cout << "Done." << std::endl;
		}

		// RMSE
		float algo_rmse = rmse(validation, algo_prediction);
		std::cout << (*i)->name() << ": \n" << algo_prediction << std::endl;
		std::cout << (*i)->name() << " RMSE: \n" << algo_rmse << std::endl;
	}
}

template <class T>
void cross_validation(const T &triplets, 
					  const typename T::value_type &max_triplet_values, 
					  bool prefer_cached_data, size_t verbosity = 0)
{
	T validation_triplets;
	T learning_triplets;
	
	if (verbosity >= 1)
	{
		std::cout << "Generating cross-validation data sets...";
	}
	cross_validation_get_sets(triplets, validation_triplets, learning_triplets, 0.1);
	if (verbosity >= 1)
	{
		std::cout << "Done." << std::endl;
	}
	
	itpp::mat learning;
	itpp::bmat learning_mask;
	learning.zeros();
	learning_mask.zeros();
	
	itpp::mat validation;
	itpp::bmat validation_mask;
	validation.zeros();
	validation_mask.zeros();

	if (verbosity >= 1)
	{
		std::cout << "Converting triplets to matrices...";
	}
	id_to_matrix_idx_converter_t users_converter(max_triplet_values.user+1);
	id_to_matrix_idx_converter_t products_converter(max_triplet_values.product+1);
	convert_triplets_to_matrix(learning, learning_mask, learning_triplets, 
							   max_triplet_values, 
							   users_converter, products_converter);
	convert_triplets_to_matrix(validation, validation_mask, validation_triplets, 
							   max_triplet_values, 
							   users_converter, products_converter);
	validation.set_size(learning.rows(), learning.cols(), true);
	//TODO: we should not, actually do this - this will take much time and memory
	//allocate right amount of memory in the right place!
	learning.set_size(users_converter.used_idxs(), 
						 products_converter.used_idxs(), true);
	learning_mask.set_size(users_converter.used_idxs(), 
						 products_converter.used_idxs(), true);
	validation.set_size(users_converter.used_idxs(), 
						 products_converter.used_idxs(), true);
	validation_mask.set_size(users_converter.used_idxs(), 
						 products_converter.used_idxs(), true);
	if (verbosity >= 1)
	{
		std::cout << "Done." << std::endl;
	}
	
	if (verbosity >= 2)
	{
		std::cout << "Learning dataset: \n" << learning << std::endl;
		std::cout << "Learning dataset mask: \n" << learning_mask << std::endl;
		std::cout << "Validation dataset: \n" << validation << std::endl;
		std::cout << "Validation dataset mask: \n" << validation_mask << std::endl;
	}

	// Users' resemblance
	itpp::mat user_resemblance(learning.rows(), learning.rows());
	itpp::bmat user_resemblance_mask(user_resemblance.rows(), user_resemblance.cols());
	user_resemblance.zeros();
	user_resemblance_mask.zeros();
	
	// compute users' resemblance on demand
	user_resemblance_itpp_t u_resemblance(learning, 
										  user_resemblance, user_resemblance_mask, 
										  correlation_coeff_resembl_metric_t());
	
	typedef collaborative_filtering_algorithm_t<itpp::mat, itpp::bmat, 
												user_resemblance_itpp_t, 
												itpp::vec> cf_itpp_algo_t;
	typedef grouplens_algo_t<itpp::mat, itpp::bmat, 
							 user_resemblance_itpp_t, 
							 itpp::vec> grouplens_itpp_algo_t;
	typedef knn_grouplens_algo_t<itpp::mat, itpp::bmat, 
								 user_resemblance_itpp_t, 
								 itpp::vec> knn_grouplens_itpp_algo_t;

	std::vector<std::shared_ptr<cf_itpp_algo_t> > algorithms;
	algorithms.push_back(std::shared_ptr<cf_itpp_algo_t>(new grouplens_itpp_algo_t));
	algorithms.push_back(std::shared_ptr<cf_itpp_algo_t>(new knn_grouplens_itpp_algo_t));

	validate_algorithms(learning, learning_mask, validation, validation_mask,
					 algorithms.begin(), algorithms.end(),
					 verbosity);
}

#endif	// SPBAU_RECOMMENDER_CROSS_VALIDATION_HPP_
