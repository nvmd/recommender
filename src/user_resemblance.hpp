
#include <cstddef>
#include <cmath>

#include <itpp/itbase.h>

// R - user's ratings of products <vector> (R[i] - rating of a product i)
// P - average ratings for products <vector> (P[i] - average rating of product i)

template <class R, class P>
float correlation_coeff(const R &user1, const R &user2, const P &avg_user_ratings)
{
	float numer = 0;
	float denom = 0;	

	float user1r_sq_sum = 0;
	float user2r_sq_sum = 0;
	
#if defined(ALG_REF_IMPL)
	//for each product
	for (size_t i=0; i<avg_prod_ratings.size(); ++i)
	{
		float user1r = user1[i] - avg_user_ratings[i];
		float user2r = user2[i] - avg_user_ratings[i];
		numer += user1r*user2r;

		user1r_sq_sum += user1r*user1r;
		user2r_sq_sum += user2r*user2r;
	}
#else	//ALG_ITPP_IMPL
	R user1r = user1 - avg_user_ratings;
	R user2r = user2 - avg_user_ratings;
	//element-wise multiplication of user1r and user2r followed by summation of resultant elements
	numer = elem_mult_sum(user1r, user2r);
	//element-wise square of user1r followed by summation of resultant elements
	user1r_sq_sum = sum_sqr(user1r);
	//element-wise square of user2r followed by summation of resultant elements
	user2r_sq_sum = sum_sqr(user2r);
#endif
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

template <class R>
class correlation_coeff_resembl_metric_t
{
public:
	correlation_coeff_resembl_metric_t(const R &avg_user_ratings)
		:m_avg_user_ratings(avg_user_ratings)
	{}
	template <class U>
	float operator()(const U &user1, const U &user2) const
	{
		return correlation_coeff(user1, user2, m_avg_user_ratings);
	}
private:
	const R &m_avg_user_ratings;
};

template <class R, class M>
void user_resembl(const R &users_ratings, R &user_resemblance, const M &user_resemblance_metric)
{
	for (int i=0; i<user_resemblance.rows(); ++i)
	{	//matrix is symmetric
		for (int j=i; j<user_resemblance.cols(); ++j)
		{
			user_resemblance(i,j) = user_resemblance_metric(users_ratings.get_row(i), users_ratings.get_row(j));
			user_resemblance(j,i) = user_resemblance(i,j);
			std::cout << "(" << i << "," << j << ") -> " << user_resemblance(i,j) << std::endl;
		}
	}
}
