
#ifndef SPBAU_RECOMMENDER_USER_RESEMBLANCE_HPP_
#define SPBAU_RECOMMENDER_USER_RESEMBLANCE_HPP_

#include <cstddef>
#include <cmath>

#include <itpp/itbase.h>

/// Correlation coefficient
/// \tparam R User's ratings of products - vector type (R[i] - rating of a product i)
/// \tparam P Average ratings for products - vector type (P[i] - average rating of product i)
/// \param[in] user1 
/// \param[in] user2 
/// \param[in] avg_user_ratings 
/// \return Correlation coefficient of 'user1' and 'user2'
template <class R, class P>
float correlation_coeff(const R &user1, const R &user2, const P &avg_user_ratings)
{
	float numer = 0;
	float denom = 0;

	float user1r_sq_sum = 0;
	float user2r_sq_sum = 0;
	
#if defined(ALG_REF_IMPL)
	//for each product
	for (int i=0; i<avg_user_ratings.size(); ++i)
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

/// User resemblance caching functor
/// \tparam RatingsT Type of the Ratings matrix
/// \tparam ResemblanceT Type of the Resemblance matrix
/// \tparam ResemblanceMaskT Type of the Resemblance matrix mask matrix
/// \tparam MetricT Type of the user resemblance metric
template <class RatingsT, class ResemblanceT, class ResemblanceMaskT, 
		  class MetricT>
class user_resemblance_t
{
public:
	/// Constructor
	/// \param[in] ratings Ratings matrix
	/// \param[in,out] resemblance Resemblance matrix
	/// \param[in,out] resemblance_mask Resemblance matrix mask matrix
	/// \param[in] metric User resemblance metric
	user_resemblance_t(const RatingsT &ratings, ResemblanceT &resemblance, 
					   ResemblanceMaskT &resemblance_mask, 
					   const MetricT &metric = MetricT())
		:m_ratings(ratings), m_metric(metric), m_resemblance(resemblance), 
		m_resemblance_mask(resemblance_mask)
	{}
	
	/// Resemblance coefficient for users
	/// \param[in] user1 First user index in the Rating matrix
	/// \param[in] user2 Second user index in the Rating matrix
	/// \return User's resemblance coefficient
	float operator()(size_t user1, size_t user2)
	{
		//std::clog << "user_resemblance_t::operator()(user1 = " << user1 << ", user2 = " << user2 << ")" << std::endl;
		if (m_resemblance_mask(user1, user2) == false)
		{
			//Note: m_resemblance matrix is symmetric - 
			// we can even store only upper triangle
			m_resemblance(user1, user2) = m_metric(m_ratings.get_row(user1), 
												   m_ratings.get_row(user2));
			m_resemblance(user2, user1) = m_resemblance(user1, user2);
			m_resemblance_mask(user1, user2) = true;
			m_resemblance_mask(user2, user1) = true;
		}
		return m_resemblance(user1, user2);
	}
	
private:
	const RatingsT &m_ratings;	///< Ratings matrix
	const MetricT &m_metric;	///< User resemblance metric
	ResemblanceT &m_resemblance;	///< Resemblance matrix
	ResemblanceMaskT &m_resemblance_mask;	///< Resemblance matrix mask matrix
};

typedef user_resemblance_t<itpp::mat, itpp::mat, itpp::bmat, 
							   correlation_coeff_resembl_metric_t<itpp::vec> > user_resemblance_itpp_t;

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

#endif	// SPBAU_RECOMMENDER_USER_RESEMBLANCE_HPP_
