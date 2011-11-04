
#include <cstddef>
#include <iostream>
#include <vector>

#include <itpp/itbase.h>
#include <itpp/base/vec.h>
#include <itpp/base/mat.h>

#include <kdtree++/kdtree.hpp>

#include "user_resemblance.hpp"
#include "grouplens.hpp"
#include "error.hpp"


#include <locale>
#include <vector>

struct csv_locale_facet : std::ctype<char>
{
	csv_locale_facet()
		: std::ctype<char>(get_table())
	{}
	static std::ctype_base::mask const* get_table()
	{
		static std::vector<std::ctype_base::mask> rc(table_size, std::ctype_base::mask());
		
		//treat characters as a whitespace
		rc[','] = std::ctype_base::space;
		rc[';'] = std::ctype_base::space;
		rc['\n'] = std::ctype_base::space;
		return &rc[0];
	}
};

struct dataset_triplet_t
{
	size_t user;
	size_t product;
	float rating;
};

template <class M, class V>
void avg_ratings(const M &users_ratings, 
					V &avg_users_rating, V &avg_product_ratings)
{
	// Average user's ratings and product's ratings
#if defined(ALG_REF_IMPL)
	for (size_t i=0; i<users_ratings.rows(); ++i)
	{
		for (size_t j=0; j<users_ratings.cols(); ++j)
		{
			avg_users_rating[i] += users_ratings(i,j);
			avg_product_ratings[j] += users_ratings(i,j);
		}
	}
#else
	for (size_t i=0; i<users_ratings.rows(); ++i)
	{
		avg_users_rating[i] = itpp::sum(users_ratings.get_row(i));
	}
	for (size_t j=0; j<users_ratings.cols(); ++j)
	{
		avg_product_ratings[j] = itpp::sum(users_ratings.get_col(j));
	}
#endif
	avg_users_rating /= avg_users_rating.size();
	avg_product_ratings /= avg_product_ratings.size();
}

template <class M, class V>
void knn(M &knn_predict, size_t k, 
			const M &users_ratings, const M &user_resemblance, 
			const V &avg_users_rating, const V &avg_product_ratings)
{
	KDTree::KDTree<3,itpp::vec> tree;
	for (size_t i=0; i<users_ratings.rows(); ++i)	//users
	{
		tree.insert(users_ratings.get_row(i));
	}
	
	for (size_t i=0; i<users_ratings.rows(); ++i)	//users
	{
		itpp::mat nearest_neighbours;
		std::vector<itpp::vec> neighbours;
		
		tree.find_within_range(users_ratings.get_row(i), k, 
				std::back_insert_iterator<std::vector<itpp::vec>>(neighbours));
		std::for_each(neighbours.begin(), neighbours.end(), 
					  [&nearest_neighbours](const itpp::vec &v){
						  nearest_neighbours.append_row(v); 
					});
		
		for (size_t j=0; j<users_ratings.cols(); ++j)	//products
		{
			knn_predict(i,j) = grouplens(avg_product_ratings, 
										nearest_neighbours, 
										avg_users_rating, i, j, 
										user_resemblance);
		}
	}
}

template <class T>
void cross_validation_get_sets(const T &triplets, T &validation, T &learning, float validation_rel_size = 0.9)
{
	size_t triplets_count = triplets.size();
	size_t validation_size = triplets_count * validation_rel_size;
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

template <class M, class T>
void convert_triplets_to_matrix(M &matrix, const T &triplets)
{
	std::for_each(triplets.begin(), triplets.end(), 
		[&matrix](const T::value_type &x){
			if (x.user > matrix.rows() || x.product > matrix.cols())
			{
				std::cout << "convert_triplets_to_matrix: resizing matrix" << std::endl;
				matrix.set_size(x.user+1, x.product+1, true);
			}
			matrix(x.user, x.product) = x.rating;
	});
}

template <class T>
void cross_validation(const T &triplets)
{
	T validation_triplets;
	T learning_triplets;
	std::cout << "Generating cross-validation data sets...";
	cross_validation_get_sets(triplets, validation_triplets, learning_triplets, 0.9);
	std::cout << "Done." << std::endl;
	
	itpp::mat learning(2176, 5636);
	itpp::mat validation(2176, 5636);
	learning.zeros();
	validation.zeros();

	std::cout << "Converting triplets to matrices...";
	convert_triplets_to_matrix(learning, learning_triplets);
	convert_triplets_to_matrix(validation, validation_triplets);
	std::cout << "Done." << std::endl;

	// Average user's and product's ratings
	std::cout << "Average user's and product's ratings...";
	itpp::vec avg_users_rating(learning.rows());
	itpp::vec avg_product_ratings(learning.cols());
	avg_users_rating.zeros();
	avg_product_ratings.zeros();
	avg_ratings(learning, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// Users' resemblance
	std::cout << "Users' resemblance...";
	itpp::mat user_resemblance(learning.rows(), learning.rows());
	user_resemblance.zeros();
	user_resembl(learning, user_resemblance, correlation_coeff_resembl_metric_t<itpp::vec>(avg_product_ratings));
	std::cout << "Done." << std::endl;
	
	// GroupLens
	std::cout << "GroupLens...";
	itpp::mat grouplens_predict(learning.rows(), learning.cols());
	grouplens_predict.zeros();
	grouplens(grouplens_predict, learning, user_resemblance, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// k-NN
	std::cout << "k-NN...";
	itpp::mat knn_predict(learning.rows(), learning.cols());
	knn_predict.zeros();
	knn(knn_predict, 3, learning, user_resemblance, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// Validation
	// validation and learning data should be of equal dimensions
	validation.set_size(learning.rows(), learning.cols(), true);

	float grouplens_rmse = rmse(validation, grouplens_predict);
	float knn_rmse = rmse(validation, knn_predict);

	// Output results
	std::cout << "Validation data: \n" << validation << std::endl;
	
	std::cout << "GroupLens: \n" << grouplens_predict << std::endl;
	std::cout << "RMSE: \n" << grouplens_rmse << std::endl;
	
	std::cout << "k-NN: \n" << knn_predict << std::endl;
	std::cout << "RMSE: \n" << knn_rmse << std::endl;
}

int main(int argc, char **argv)
{
	if (setvbuf(stdout, NULL, _IONBF, 0) != 0)	// unbuffered
	//if (setvbuf(stdout, NULL, _IOLBF, 0) != 0)	// line buffering of stdout
	{
		perror("setvbuf");
	}
	
	std::cout << "Reading dataset...";
	std::cin.imbue(std::locale(std::locale(), new csv_locale_facet()));
	std::vector<dataset_triplet_t> triplet_list;
	triplet_list.reserve(3000);
	dataset_triplet_t triplet;
	while (std::cin >> triplet.user >> triplet.product >> triplet.rating)
	{
		std::cout << "(" << triplet.user << "," << triplet.product << ") -> " << triplet.rating << std::endl;
		triplet_list.push_back(triplet);
	}
	std::cout << "Done." << std::endl;

	cross_validation(triplet_list);

	return 0;
}
