
#include <cstddef>
#include <cmath>

template <class V, class M, class R>
float grouplens(const V &avg_product_rating, // avg_product_rating[i] - average user's rating of a product 'i'
			const M &users_rating,	// users_rating[i][j] - 'i' user's rating of a product 'j'
			const V &avg_users_rating,	// avg_users_rating[i] - average rating of the user 'i'
			size_t user,
			size_t product,
			R resemblance)
{
	float numer = 0;
	float denom = 0;
	
	for (size_t i=0; i<users_rating.rows(); ++i)
	{
		//float user_resemblance = correlation_coeff(users_rating.get_row(user),users_rating.get_row(i),avg_product_rating);
		float user_resemblance = resemblance(users_rating.get_row(user),users_rating.get_row(i),avg_product_rating);
		
		numer += (users_rating.get_row(i)[product] - avg_users_rating[i])*user_resemblance;
		denom += std::abs(user_resemblance);
	}
	
	return avg_product_rating[product] + (numer/denom);
}
