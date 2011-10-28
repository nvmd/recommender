
#include <cstddef>
#include <cmath>

// R - user's ratings of products <vector> (R[i] - rating of a product i)
// P - average ratings for products <vector> (P[i] - average rating of product i)

template <class R, class P>
float correlation_coeff(const R &user1, const R &user2, const P &avg_prod_ratings)
{
	float numer = 0;
	float denom = 0;	

	float user1r_sq_sum = 0;
	float user2r_sq_sum = 0;
	
	// for each product
	for (size_t i=0; i<avg_prod_ratings.size(); ++i)
	{
		float user1r = user1[i] - avg_prod_ratings[i];
		float user2r = user2[i] - avg_prod_ratings[i];
		numer += user1r*user2r;

		user1r_sq_sum += user1r*user1r;
		user2r_sq_sum += user2r*user2r;
	}
	
	denom = sqrt(user1r_sq_sum)*sqrt(user2r_sq_sum);

	return numer/denom;
}

template <class R, class P>
float cosine_angle(const R &user1, const R &user2, const P &avg_prod_ratings)
{
	float numer = 0;
	float denom = 0;
	
	float user1r_sq_sum = 0;
	float user2r_sq_sum = 0;

	// for each product
	for (size_t i=0; i<avg_prod_ratings.size(); ++i)
	{
		numer += user1[i]*user2[i];
		
		user1r_sq_sum += user1[i]*user1[i];
		user2r_sq_sum += user2[i]*user2[i];
	}

	denom = sqrt(user1r_sq_sum)*sqrt(user2r_sq_sum);

	return numer/denom;
}
