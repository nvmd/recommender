
#include <cstddef>
#include <iostream>
#include <list>

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
	// k-NN
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

int main(int argc, char **argv)
{
	if (setvbuf(stdout, NULL, _IONBF, 0) != 0)	// unbuffered
	//if (setvbuf(stdout, NULL, _IOLBF, 0) != 0)	// line buffering of stdout
	{
		perror("setvbuf");
	}
	
	std::cout << "Reading dataset...";
	std::cin.imbue(std::locale(std::locale(), new csv_locale_facet()));
	std::list<dataset_triplet_t> triplet_list;
	dataset_triplet_t triplet;
	while (std::cin >> triplet.user >> triplet.product >> triplet.rating)
	{
		std::cout << "(" << triplet.user << "," << triplet.product << ") -> " << triplet.rating << std::endl;
		triplet_list.push_back(triplet);
	}
	std::cout << "Done." << std::endl;

	itpp::mat users_ratings(2176,5636);
	users_ratings.zeros();
	itpp::it_file users_ratings_file("users_ratings.it");
	users_ratings_file << itpp::Name("users_ratings") << users_ratings;
	
	std::cout << "Average user's and product's ratings...";
	itpp::vec avg_users_rating(users_ratings.rows());
	itpp::vec avg_product_ratings(users_ratings.cols());
	avg_users_rating.zeros();
	avg_product_ratings.zeros();
	avg_ratings(users_ratings, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	itpp::it_file avg_users_rating_file("avg_users_rating.it");
	avg_users_rating_file << itpp::Name("avg_users_rating") << avg_users_rating;
	itpp::it_file avg_product_ratings_file("avg_product_ratings.it");
	avg_product_ratings_file << itpp::Name("avg_product_ratings") << avg_product_ratings;
	
	// Users' resemblance
	std::cout << "Users' resemblance...";
	itpp::mat user_resemblance(users_ratings.rows(), users_ratings.rows());
	user_resemblance.zeros();
	user_resembl(users_ratings, user_resemblance, correlation_coeff_resembl_metric_t<itpp::vec>(avg_product_ratings));
	std::cout << "Done." << std::endl;
	itpp::it_file user_resemblance_file("user_resemblance.it");
	user_resemblance_file << itpp::Name("user_resemblance") << user_resemblance;
	
	// GroupLens
	std::cout << "GroupLens...";
	itpp::mat grouplens_predict(users_ratings.rows(), users_ratings.cols());
	grouplens_predict.zeros();
	grouplens(grouplens_predict, users_ratings, user_resemblance, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// k-NN
	std::cout << "k-NN...";
	itpp::mat knn_predict(users_ratings.rows(), users_ratings.cols());
	knn_predict.zeros();
	knn(knn_predict, 3, users_ratings, user_resemblance, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// Output results
	itpp::mat realdata(users_ratings);
	std::cout << "Real data: \n" << realdata << std::endl;
	
	std::cout << "GroupLens: \n" << grouplens_predict << std::endl;
	std::cout << "RMSE: \n" << rmse(grouplens_predict, realdata) << std::endl;
	
	std::cout << "k-NN: \n" << knn_predict << std::endl;
	std::cout << "RMSE: \n" << rmse(knn_predict, realdata) << std::endl;
	
	return 0;
}
