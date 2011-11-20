
#ifndef SPBAU_RECOMMENDER_KNN_HPP_
#define SPBAU_RECOMMENDER_KNN_HPP_

#include <vector>
#include <iostream>

#include <itpp/base/mat.h>
#include <kdtree++/kdtree.hpp>

template <class M>
class kdtree_distance_t
{
public:
	typedef float distance_type;
	
	kdtree_distance_t(M &resemblance)
		:m_resemblance(resemblance)
	{}
	
	distance_type operator()(size_t user1, size_t user2) const
	{
		distance_type c = m_resemblance(user1, user2);
		return c*c;
	}
	
private:
	M &m_resemblance;
};

template <class M>
class user_rating_accessor_t
{
public:
	typedef typename M::value_type result_type;
	
	user_rating_accessor_t(const M &users_ratings)
		:m_users_ratings(users_ratings)
	{}
	
	result_type operator()(size_t const& v, size_t const n) const
	{
		return m_users_ratings.get_row(v)[n];
	}
	
private:
	const M &m_users_ratings;
};

template <class M, class V, class R>
void knn(M &knn_predict, size_t k, 
			const M &users_ratings, R &user_resemblance, 
			const V &avg_users_rating, const V &avg_product_ratings)
{
	KDTree::KDTree<3, 
				size_t, 
				user_rating_accessor_t<M>, //access i-th element of the vector (using operator[]) (result_type operator()(_Val const& V, size_t const N) const)
				kdtree_distance_t<R>	//squared distance between vectors (distance_type operator() (const _Tp& __a, const _Tp& __b) const)
				> 
				tree(user_rating_accessor_t<M>(users_ratings), kdtree_distance_t<R>(user_resemblance));
	for (int i=0; i<users_ratings.rows(); ++i)	//users
	{
		tree.insert(i);
	}
	
	for (int i=0; i<users_ratings.rows(); ++i)	//users
	{
		itpp::mat nearest_neighbours;
		std::vector<size_t> neighbours;
		
		// Nearest neighbours of the i-th user
		std::cout << "neighbours of " << i << " (" << users_ratings.get_row(i) << ")" << std::endl;
		tree.find_within_range(i, k, 
				std::back_insert_iterator<std::vector<size_t>>(neighbours));
		std::for_each(neighbours.begin(), neighbours.end(), 
					  [&nearest_neighbours,&users_ratings](const size_t &v){
						  itpp::vec neighbour(users_ratings.get_row(v));
						  std::cout << "neighbour: " << v << " : " << neighbour << std::endl;
						  nearest_neighbours.append_row(neighbour);
					});
		
		//assert(users_ratings.cols() == nearest_neighbours.cols());
		// Estimate i-th user by its nearest neighbours using GroupLens
		for (int j=0; j<users_ratings.cols(); ++j)	//products
		{
			knn_predict(i,j) = grouplens(avg_product_ratings, 
										nearest_neighbours, 
										avg_users_rating, i, j, 
										user_resemblance);
		}
		std::cout << "neareast neighbours of " << i 
				  << " " << users_ratings.get_row(i) << ": " 
				  << nearest_neighbours << std::endl;
		std::cout << "knn_predict: " << knn_predict << std::endl;
	}
}

#endif	// SPBAU_RECOMMENDER_KNN_HPP_
