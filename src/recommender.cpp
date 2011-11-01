
#include <cstddef>
#include <iostream>

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

int main(int argc, char **argv)
{
	itpp::mat users_ratings(2176,5636);
	users_ratings.zeros();
	
	std::cout << "Reading dataset...";
	size_t user;
	size_t product;
	float rating;
	if (setvbuf(stdout, NULL, _IONBF, 0) != 0)	// unbuffered
	//if (setvbuf(stdout, NULL, _IOLBF, 0) != 0)	// line buffering of stdout
	{
		perror("setvbuf");
	}
	std::cin.imbue(std::locale(std::locale(), new csv_locale_facet()));
	while (std::cin >> user >> product >> rating)
	{
		std::cout << "(" << user << "," << product << ") -> " << rating << std::endl;
		if (user >= users_ratings.rows() || product >= users_ratings.cols())
		{
			std::cout << "Resizing user's ratings matrix...";
			users_ratings.set_size(user+1, product+1, true);
			std::cout << "Done." << std::endl;
		}
		users_ratings(user,product) = rating;
	}
	std::cout << "Done." << std::endl;
	itpp::it_file users_ratings_file("users_ratings.it");
	users_ratings_file << itpp::Name("users_ratings") << users_ratings;
	
	std::cout << "Average user's and product's ratings...";
	itpp::vec avg_users_rating(users_ratings.rows());
	itpp::vec avg_product_ratings(users_ratings.cols());
	avg_users_rating.zeros();
	avg_product_ratings.zeros();
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
	std::cout << "Done." << std::endl;
	
	itpp::it_file avg_users_rating_file("avg_users_rating.it");
	avg_users_rating_file << itpp::Name("avg_users_rating") << avg_users_rating;
	itpp::it_file avg_product_ratings_file("avg_product_ratings.it");
	avg_product_ratings_file << itpp::Name("avg_product_ratings") << avg_product_ratings;
	
	itpp::mat user_resemblance(users_ratings.rows(),users_ratings.rows());
	user_resemblance.zeros();
	std::cout << "User resemblance...";
	for (size_t i=0; i<user_resemblance.rows(); ++i)
	{	//matrix is symmetric
		for (size_t j=i; j<user_resemblance.cols(); ++j)
		{
			user_resemblance(i,j) = correlation_coeff(users_ratings.get_row(i), 
													  users_ratings.get_row(j), 
													  avg_product_ratings);
			user_resemblance(j,i) = user_resemblance(i,j);
			std::cout << "(" << i << "," << j << ") -> " << user_resemblance(i,j) << std::endl;
		}
	}
	std::cout << "Done." << std::endl;
	itpp::it_file user_resemblance_file("user_resemblance.it");
	user_resemblance_file << itpp::Name("user_resemblance") << user_resemblance;
	
	std::cout << "GroupLens...";
	// GroupLens
	itpp::mat grouplens_predict(users_ratings.rows(),users_ratings.cols());
	for (size_t i=0; i<users_ratings.rows(); ++i)	// users
	{
		for (size_t j=0; j<users_ratings.cols(); ++j)	// products
		{
			grouplens_predict(i,j) = grouplens(avg_product_ratings, 
										users_ratings, 
										avg_users_rating, i, j, 
										user_resemblance);
		}
	}
	std::cout << "Done." << std::endl;
	
	std::cout << "k-NN...";
	// k-NN
	KDTree::KDTree<3,itpp::vec> tree;
	for (size_t i=0; i<users_ratings.rows(); ++i)	//users
	{
		tree.insert(users_ratings.get_row(i));
	}
	
	itpp::mat knn_predict(users_ratings.rows(),users_ratings.cols());
	for (size_t i=0; i<users_ratings.rows(); ++i)	//users
	{
		itpp::mat nearest_neighbours;
		std::vector<itpp::vec> neighbours;
		
		tree.find_within_range(users_ratings.get_row(i), 3, 
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
	std::cout << "Done." << std::endl;
	
	itpp::mat realdata(users_ratings);
	std::cout << "Real data: \n" << realdata << std::endl;
	
	std::cout << "GroupLens: \n" << grouplens_predict << std::endl;
	std::cout << "RMSE: \n" << rmse(grouplens_predict, realdata) << std::endl;
	
	std::cout << "k-NN: \n" << knn_predict << std::endl;
	std::cout << "RMSE: \n" << rmse(knn_predict, realdata) << std::endl;
	
	return 0;
}
