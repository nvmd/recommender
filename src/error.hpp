
#include <cmath>
#include <cstddef>

// R - rating
template <class R>
float rmse(const R &prediction, const R &real)
{
	float sum = 0;
	for (size_t i = 0; i < prediction.cols(); ++i)
	{
		sum += (real.get_col(i)-prediction.get_col(i))*(real.get_col(i)-prediction.get_col(i));
	}
	return sqrt((1.0/prediction.cols()) * sum);
}
