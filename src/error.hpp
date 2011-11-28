
#ifndef SPBAU_RECOMMENDER_ERROR_HPP_
#define SPBAU_RECOMMENDER_ERROR_HPP_

#include <cmath>
#include <cstddef>
#include <cassert>

/// Matrix RMSE
/// \tparam R Matrix type
/// \param[in] real First argument matrix
/// \param[in] prediction Second argument matrix
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
/// \tparam V Vector type
/// \param[in] x1 First argument vector
/// \param[in] x2 Second argument vector
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

/// Matrix RMSE
/// \tparam R Matrix type
/// \tparam M Matrix mask type
/// \param[in] real First argument matrix
/// \param[in] real_mask First argument vector mask
/// \param[in] prediction Second argument matrix
/// \param[in] prediction_mask Second argument vector mask
/// \return RMSE of two matrices
template <class R, class M>
float rmse(const R &real, const M &real_mask, 
		   const R &prediction, const M &prediction_mask)
{
	assert(real.cols() == prediction.cols() 
		   && real.rows() == prediction.rows());
	assert(real_mask.cols() == prediction_mask.cols()
		   && real_mask.rows() == prediction_mask.rows());
	assert(real.cols() == real_mask.cols()
		   && real.rows() == real_mask.rows());

	float sum = 0;
	size_t valid_cols = 0;
	for (int i = 0; i < prediction.cols(); ++i)
	{
		float valid_elems = 0;
		float rmse_result = rmse_v(real.get_col(i), real_mask.get_col(i), 
					  prediction.get_col(i), prediction_mask.get_col(i), 
					  valid_elems);
		if (valid_elems > 0)
		{
			sum += rmse_result;
			++valid_cols;
		}
	}
	
	return std::sqrt((1.0/valid_cols) * sum);
}

/// Vector RMSE
/// \tparam V Vector type
/// \tparam M Vector mask type
/// \param[in] x1 First argument vector
/// \param[in] x1_mask First argument vector mask
/// \param[in] x2 Second argument vector
/// \param[in] x2_mask Second argument vector mask
/// \param[out] valid_elems Number of elements which are valid in both vectors
/// \return RMSE of two vectors
template <class V, class M>
float rmse_v(const V &x1, const M &x1_mask, 
			 const V &x2, const M &x2_mask,
			 size_t &valid_elems)
{
	assert(x1.size() == x2.size());
	assert(x1_mask.size() == x2_mask.size());
	assert(x1_mask.size() == x1.size());

	float sum = 0;
	/*size_t*/ valid_elems = 0;
	for (int i = 0; i < x1.size(); ++i)
	{
		if (x1_mask[i] && x2_mask[i])
		{
			sum += (x1[i] - x2[i])*(x1[i] - x2[i]);
			++valid_elems;
		}
	}
	return valid_elems > 0 ? std::sqrt((1.0/valid_elems) * sum) : 0;
}

#endif	// SPBAU_RECOMMENDER_ERROR_HPP_
