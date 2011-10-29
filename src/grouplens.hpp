
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
	
	for (size_t i=0; i<users_rating.rows(); ++i)
	{
		float user_resemblance = resemblance(user,i);
		
		numer += (users_rating.get_row(i)[product] - avg_users_rating[i])*user_resemblance;
		denom += std::abs(user_resemblance);
	}
	
	return avg_product_rating[product] + (numer/denom);
}
