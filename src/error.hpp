
#ifndef SPBAU_RECOMMENDER_ERROR_HPP_
#define SPBAU_RECOMMENDER_ERROR_HPP_

#include <cmath>
#include <cstddef>
#include <cassert>

/// Matrix RMSE
/// \param R Matrix type
/// \return RMSE of two matrices
template <class R>
float rmse(const R &real, const R &prediction)
{
	assert(real.cols() == prediction.cols() 
		&& real.rows() == prediction.rows());

	float sum = 0;
#if defined(ALG_REF_IMPL)
	for (int i = 0; i < prediction.cols(); ++i)
	{
		sum += rmse_v(real.get_col(i), prediction.get_col(i));
	}
#else	// ALG_ITPP_IMPL
	for (int i = 0; i < prediction.cols(); ++i)
	{
		sum += sum_sqr(real.get_col(i) - prediction.get_col(i));
	}
#endif
	return std::sqrt((1.0/prediction.cols()) * sum);
}

/// Vector RMSE
/// \param V Vector type
/// \param x1 First argument vector
/// \param x2 Second argument vector
/// \return RMSE of two vectors
template <class V>
float rmse_v(const V &x1, const V &x2)
{
	assert(x1.size() == x2.size());

	float sum = 0;
#if defined(ALG_REF_IMPL)
	for (int i = 0; i < x1.size(); ++i)
	{
		sum += (x1(i) - x2(i))*(x1(i) - x2(i));
	}
#else	// ALG_ITPP_IMPL
	sum = sum_sqr(x1 - x2);
#endif
	return std::sqrt((1.0/x1.size()) * sum);
}

#endif	// SPBAU_RECOMMENDER_ERROR_HPP_
