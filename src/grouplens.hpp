
#ifndef SPBAU_RECOMMENDER_GROUPLENS_HPP_
#define SPBAU_RECOMMENDER_GROUPLENS_HPP_

#include <cstddef>
#include <cmath>

template <class V, class M>
float grouplens(const V &avg_product_rating, // avg_product_rating[i] - average user's rating of a product 'i'
			const M &users_rating,	// users_rating[i][j] - 'i' user's rating of a product 'j'
			const V &avg_users_rating,	// avg_users_rating[i] - average rating of the user 'i'
			size_t user,
			size_t product,
			const M &resemblance)
{
	float numer = 0;
	float denom = 0;
	
	for (int i=0; i<users_rating.rows(); ++i)
	{
		float user_resemblance = resemblance(user,i);
		
		numer += (users_rating.get_row(i)[product] - avg_users_rating[i])*user_resemblance;
		denom += std::abs(user_resemblance);
	}
	
	return avg_product_rating[product] + (numer/denom);
}

template <class M, class V>
void grouplens(M &grouplens_predict, const M &users_ratings, 
				const M &user_resemblance, 
				const V &avg_users_rating, const V &avg_product_ratings)
{
	for (int i=0; i<users_ratings.rows(); ++i)	// users
	{
		for (int j=0; j<users_ratings.cols(); ++j)	// products
		{
			grouplens_predict(i,j) = grouplens(avg_product_ratings, 
										users_ratings, 
										avg_users_rating, i, j, 
										user_resemblance);
		}
	}
}

#endif	// SPBAU_RECOMMENDER_GROUPLENS_HPP_
