
#ifndef SPBAU_RECOMMENDER_GROUPLENS_HPP_
#define SPBAU_RECOMMENDER_GROUPLENS_HPP_

#include <cstddef>
#include <cmath>

/// GroupLens
/// \param[in] avg_product_rating Average rating of the product (user's rating of a product)
/// \param[in] users_rating User-Product rating matrix
/// \param[in] avg_users_rating Average rating of the user (rating of the user)
/// \param[in] user User index in the rating matrix
/// \param[in] product Product index in the rating matrix
/// \param[in,out] resemblance Resemblance matrix (can be modified if lazy computation is used)
/// \return Predicted 'product' rating by the 'user'
template <class V, class M, class R>
float grouplens(const V &avg_product_rating, const M &users_rating,
				const V &avg_users_rating, 
				size_t user, size_t product, R &resemblance)
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

template <class M, class V, class R>
void grouplens(M &grouplens_predict, const M &users_ratings, 
				R &user_resemblance, 
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
