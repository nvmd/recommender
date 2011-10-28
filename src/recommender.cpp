
#include <cstddef>
#include <iostream>

#include <itpp/base/vec.h>
#include <itpp/base/mat.h>
#include <kdtree++/kdtree.hpp>

#include "user_resemblance.hpp"
#include "grouplens.hpp"
#include "error.hpp"

int main(int argc, char **argv)
{
	itpp::mat users_ratings;
	itpp::vec avg_users_rating(users_ratings.rows());
	itpp::vec avg_product_ratings(users_ratings.cols());
	itpp::mat realdata(users_ratings);
	
	
	// Average user's ratings and product's ratings
	for (size_t i=0; i<users_ratings.rows(); ++i)
	{
		for (size_t j=0; j<users_ratings.cols(); ++i)
		{
			avg_users_rating[i] += users_ratings(i,j);
			avg_product_ratings[j] += users_ratings(i,j);
		}
	}
	
	// GroupLens
	itpp::mat grouplens_predict(users_ratings.rows(),users_ratings.cols());
	for (size_t i=0; i<users_ratings.rows(); ++i)	// users
	{
		for (size_t j=0; j<users_ratings.cols(); ++j)	// products
		{
			grouplens_predict(i,j) = grouplens(avg_product_ratings, 
										users_ratings, 
										avg_users_rating, i, j, 
										correlation_coeff<itpp::vec,itpp::vec>);
		}
	}
	
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
		
		tree.find_within_range(users_ratings.get_row(i), 10, 
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
										correlation_coeff<itpp::vec,itpp::vec>);
		}
	}
	
	std::cout << "Real data: \n" << realdata << std::endl;
	
	std::cout << "GroupLens: \n" << grouplens_predict << std::endl;
	std::cout << "RMSE: \n" << rmse(grouplens_predict, realdata) << std::endl;
	
	std::cout << "k-NN: \n" << knn_predict << std::endl;
	std::cout << "RMSE: \n" << rmse(knn_predict, realdata) << std::endl;
	
	return 0;
}
