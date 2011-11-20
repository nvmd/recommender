
#ifndef SPBAU_RECOMMENDER_KNN_HPP_
#define SPBAU_RECOMMENDER_KNN_HPP_

#include <vector>
#include <iostream>

#include <itpp/base/mat.h>
#include <kdtree++/kdtree.hpp>

template <class R, class M>
class kdtree_distance_t
{
public:
	kdtree_distance_t(const R &avg_user_ratings, M &resemblance)
		:m_avg_user_ratings(avg_user_ratings), 
		m_resemblance(resemblance)
	{}
	template <class U>
	float operator()(const U &user1, const U &user2) const
	{
		float c = 0;
		if (!false)	//TODO: if not present in 'm_resemblance' matrix
		{
			c = correlation_coeff(user1, user2, m_avg_user_ratings);
			m_resemblance(user1, user2) = c;
			m_resemblance(user2, user1) = c;
		}
		else
		{
			c = m_resemblance(user1, user2);
		}
		return c*c;
	}
	typedef float distance_type;
private:
	const R &m_avg_user_ratings;
	M &m_resemblance;
};

template <class M, class V, class R, class B>
void knn(M &knn_predict, size_t k, 
		 const M &users_ratings, const B &users_ratings_mask, 
		 R &user_resemblance, 
		 const V &avg_users_rating, const V &avg_product_ratings)
{
	//M user_resemblance(user_resemblance_unused.rows(), user_resemblance_unused.cols());
	//M user_resemblance(user_resemblance_unused);
	//user_resemblance.zeros();
	KDTree::KDTree<3, 
				itpp::vec, 
				KDTree::_Bracket_accessor<itpp::vec>, //access i-th element of the vector (using operator[]) (result_type operator()(_Val const& V, size_t const N) const)
				kdtree_distance_t<itpp::vec, itpp::mat>	//squared distance between vectors (distance_type operator() (const _Tp& __a, const _Tp& __b) const)
				> 
				tree(KDTree::_Bracket_accessor<itpp::vec>(), 
					kdtree_distance_t<itpp::vec, itpp::mat>(avg_users_rating, user_resemblance));
	for (int i=0; i<users_ratings.rows(); ++i)	//users
	{
		tree.insert(users_ratings.get_row(i));
	}
	
	for (int i=0; i<users_ratings.rows(); ++i)	//users
	{
		itpp::mat nearest_neighbours;
		std::vector<itpp::vec> neighbours;
		
		// Nearest neighbours of the i-th user
		std::cout << "neighbours of " << i << " (" << users_ratings.get_row(i) << ")" << std::endl;
		tree.find_within_range(users_ratings.get_row(i), k, 
				std::back_insert_iterator<std::vector<itpp::vec>>(neighbours));
		std::for_each(neighbours.begin(), neighbours.end(), 
					  [&nearest_neighbours](const itpp::vec &v){
						  std::cout << "neighbour: " << v << std::endl;
						  nearest_neighbours.append_row(v);
					});
		
		//assert(users_ratings.cols() == nearest_neighbours.cols());
		// Estimate i-th user by its nearest neighbours using GroupLens
		for (int j=0; j<users_ratings.cols(); ++j)	//products
		{
			if (users_ratings_mask(i,j) == false)
			{
				knn_predict(i,j) = grouplens(avg_product_ratings, 
											nearest_neighbours, 
											avg_users_rating, i, j, 
											user_resemblance);
			}
		}
		std::cout << "neareast neighbours of " << i << ": " << nearest_neighbours << std::endl;
		std::cout << "knn_predict: " << knn_predict << std::endl;
	}
}

#endif	// SPBAU_RECOMMENDER_KNN_HPP_
