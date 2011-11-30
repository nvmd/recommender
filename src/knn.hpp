
#ifndef SPBAU_RECOMMENDER_KNN_HPP_
#define SPBAU_RECOMMENDER_KNN_HPP_

#include <vector>
#include <iostream>
#include <algorithm>
#include <utility>

#include <itpp/base/mat.h>
#include <kdtree++/kdtree.hpp>

#include "user_resemblance.hpp"

class kdtree_distance_cosine_angle_t
{
public:
	kdtree_distance_cosine_angle_t()
	{}
	template <class U>
	float operator()(const U &user1, const U &user2) const
	{
		float c = 0;
		c = cosine_angle(user1, user2);
		return c*c;
	}
	typedef float distance_type;
};

class kdtree_distance_correlation_coeff_t
{
public:
	kdtree_distance_correlation_coeff_t()
	{}
	template <class U>
	float operator()(const U &user1, const U &user2) const
	{
		float c = 0;
		c = correlation_coeff(user1, user2);
		return c*c;
	}
	typedef float distance_type;
};

template <class T, class V, class OutputIterator>
void k_nearest(T &tree, const V &value, size_t k, OutputIterator out)
{
	size_t found = 0;
	std::vector<V> neighbours;
	neighbours.reserve(k+1);
	while (found < k)
	{
		std::pair<typename T::const_iterator, typename T::distance_type> nearest = tree.find_nearest(value);
 		if (nearest.first == tree.end())
 			return;
		neighbours.push_back(*(nearest.first));
		++out = nearest.first;
		++found;
		tree.erase_exact(nearest.first);
	}
	std::for_each(neighbours.begin(), neighbours.end(), 
				  [&tree](const V &x) {
					tree.insert(x);
				  });
	tree.optimise();
}

template <class M, class V, class R, class B>
void knn(M &knn_predict, double k, 
		 const M &users_ratings, const B &users_ratings_mask, 
		 R &user_resemblance, 
		 const V &avg_users_rating, const V &avg_product_ratings)
{
	typedef itpp::vec kdtree_value_type;
	//access i-th element of the vector (using operator[]) (result_type operator()(_Val const& V, size_t const N) const)
	typedef KDTree::_Bracket_accessor<itpp::vec> kdtree_bracket_accessor_type;
	//squared distance between vectors (distance_type operator() (const _Tp& __a, const _Tp& __b) const)
	typedef kdtree_distance_correlation_coeff_t kdtree_distance_type;
	typedef KDTree::KDTree<kdtree_value_type, 
						   kdtree_bracket_accessor_type, 
						   kdtree_distance_type> kdtree_type;
	
	kdtree_bracket_accessor_type kdtree_bracket_accessor;
	kdtree_distance_type kdtree_distance;
	
	kdtree_type tree(users_ratings.cols(), kdtree_bracket_accessor, kdtree_distance);
	for (int i=0; i<users_ratings.rows(); ++i)	//users
	{
		tree.insert(users_ratings.get_row(i));
	}
	
	for (int i=0; i<users_ratings.rows(); ++i)	//users
	{
		itpp::mat nearest_neighbours;
		std::vector<kdtree_value_type> neighbours;
		
		// Nearest neighbours of the i-th user
#define KNN_WITHIN_RANGE
#if defined(KNN_WITHIN_RANGE)
		std::cout << "\nNeighbours of " << i << " (" << users_ratings.get_row(i) << ") within " << k << ": " << std::endl;
		tree.find_within_range(users_ratings.get_row(i), k, 
				std::back_insert_iterator<std::vector<kdtree_value_type>>(neighbours));
#else
		std::cout << "\n" << k << " nearest neighbours of " << i << " (" << users_ratings.get_row(i) << "): " << std::endl;
		k_nearest(tree, users_ratings.get_row(i), k, 
				  std::back_insert_iterator<std::vector<kdtree_value_type>>(neighbours));
#endif	
		std::for_each(neighbours.begin(), neighbours.end(), 
					  [&nearest_neighbours,&i,&users_ratings,&kdtree_distance](const kdtree_value_type &v){
						  std::cout << "\t" << v << " at distance " << kdtree_distance(users_ratings.get_row(i), v) << std::endl;
						  nearest_neighbours.append_row(v);
					});
		
		//assert(users_ratings.cols() == nearest_neighbours.cols());
		// Estimate i-th user by its nearest neighbours using GroupLens
		for (int j = 0; j < users_ratings.cols(); ++j)	//products
		{
			if (users_ratings_mask(i,j) == false)
			{
				knn_predict(i,j) = grouplens(avg_product_ratings, 
											nearest_neighbours, 
											avg_users_rating, i, j, 
											user_resemblance);
			}
		}
	}
}

#endif	// SPBAU_RECOMMENDER_KNN_HPP_
