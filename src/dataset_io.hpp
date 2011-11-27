
#ifndef SPBAU_RECOMMENDER_DATASET_IO_HPP_
#define SPBAU_RECOMMENDER_DATASET_IO_HPP_

#include <locale>
#include <vector>
#include <algorithm>
#include <iostream>

#include <itpp/base/vec.h>

struct csv_locale_facet : std::ctype<char>
{
	csv_locale_facet()
		: std::ctype<char>(get_table())
	{}
	static std::ctype_base::mask const* get_table()
	{
		static std::vector<std::ctype_base::mask> rc(table_size, std::ctype_base::mask());
		
		//treat characters as a whitespace
		rc[','] = std::ctype_base::space;
		rc[';'] = std::ctype_base::space;
		rc['\n'] = std::ctype_base::space;
		rc['\r'] = std::ctype_base::space;
		return &rc[0];
	}
};

struct dataset_triplet_t
{
	size_t user;	///< User ID
	size_t product;	///< Product ID
	float rating;	///< Product rating
};

class id_to_matrix_idx_converter_t
{
public:
	id_to_matrix_idx_converter_t(size_t table_size)
		:m_conversion_table(table_size), m_used(0)
	{
		for (size_t i = 0; i < table_size; ++i)
		{
			m_conversion_table[i] = -1;
		}
	}
	size_t operator()(size_t id)
	{
		if (m_conversion_table[id] == -1)
		{
			m_conversion_table[id] = m_used++;
		}
		return m_conversion_table[id];
	}
	size_t used_idxs() const
	{
		return m_used;
	}
private:
	itpp::ivec m_conversion_table;
	size_t m_used;
};

template <class M, class B, class T>
void convert_triplets_to_matrix(M &matrix, B &matrix_mask, const T &triplets, 
								const typename T::value_type &max_triplet_values,
								id_to_matrix_idx_converter_t &users_converter,
								id_to_matrix_idx_converter_t &products_converter)
{
	matrix.set_size(max_triplet_values.user+1, max_triplet_values.product+1);
	matrix_mask.set_size(matrix.rows(), matrix.cols());
	matrix.zeros();
	matrix_mask.zeros();

	std::for_each(triplets.begin(), triplets.end(), 
		[&](const typename T::value_type &x){
			size_t user = users_converter(x.user);
			size_t product = products_converter(x.product);
			std::cout << "(" << x.user << ", " << x.product << ") -> (" 
							 << user << ", " << product << ")" << std::endl;
			
			if (user >= static_cast<size_t>(matrix.rows()) 
				|| product >= static_cast<size_t>(matrix.cols()))
			{
				std::cout << "convert_triplets_to_matrix: resizing matrix" 
						  << std::endl;
				matrix.set_size(user+1, product+1, true);
				matrix_mask.set_size(user+1, product+1, true);
			}
			matrix(user, product) = x.rating;
			matrix_mask(user, product) = true;
	});
}

template <class F, class L, class T>
void read_dataset(F &file, L &triplet_list, T &max_triplet_values, size_t input_limit = 0, size_t skip_lines = 0)
{
	file.imbue(std::locale(std::locale(), new csv_locale_facet()));
	dataset_triplet_t triplet = {0, 0, 0};
	
	size_t lines_skipped = 0;
	std::string line;
	while (lines_skipped++ < skip_lines)
	{
		std::getline(file, line);
		std::cout << "Skipped line: \"" << line << "\"" << std::endl;
	}
	
	while ((input_limit == 0 || triplet_list.size() < input_limit) 
			&& file >> triplet.user >> triplet.product >> triplet.rating)
	{
		std::cout << "(" << triplet.user << ", " << triplet.product << ") -> " 
						 << triplet.rating << std::endl;

		max_triplet_values.user = std::max(max_triplet_values.user, triplet.user);
		max_triplet_values.product = std::max(max_triplet_values.product, triplet.product);
		triplet_list.push_back(triplet);
	}
}

#endif	// SPBAU_RECOMMENDER_DATASET_IO_HPP_
