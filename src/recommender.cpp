
#include <cstddef>
#include <iostream>
#include <vector>

#include <itpp/itbase.h>
#include <itpp/base/vec.h>
#include <itpp/base/mat.h>

#include <kdtree++/kdtree.hpp>
#include <tclap/CmdLine.h>

#include "user_resemblance.hpp"
#include "grouplens.hpp"
#include "error.hpp"


#include <locale>
#include <vector>

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
		return &rc[0];
	}
};

struct dataset_triplet_t
{
	size_t user;
	size_t product;
	float rating;
};

template <class M, class V>
void avg_ratings(const M &users_ratings, 
					V &avg_users_rating, V &avg_product_ratings)
{
	// Average user's ratings and product's ratings
#if defined(ALG_REF_IMPL)
	for (size_t i=0; i<users_ratings.rows(); ++i)
	{
		for (size_t j=0; j<users_ratings.cols(); ++j)
		{
			avg_users_rating[i] += users_ratings(i,j);
			avg_product_ratings[j] += users_ratings(i,j);
		}
	}
#else
	for (size_t i=0; i<users_ratings.rows(); ++i)
	{
		avg_users_rating[i] = itpp::sum(users_ratings.get_row(i));
	}
	for (size_t j=0; j<users_ratings.cols(); ++j)
	{
		avg_product_ratings[j] = itpp::sum(users_ratings.get_col(j));
	}
#endif
	avg_users_rating /= avg_users_rating.size();
	avg_product_ratings /= avg_product_ratings.size();
}

template <class R, class M>
class kdtree_distance_t
{
public:
	kdtree_distance_t(const R &avg_product_ratings, M &resemblance)
		:m_avg_product_ratings(avg_product_ratings), 
		m_resemblance(resemblance)
	{}
	template <class U>
	float operator()(const U &user1, const U &user2) const
	{
		float c = 0;
		if (!false)	//TODO: if not present in 'm_resemblance' matrix
		{
			c = correlation_coeff(user1, user2, m_avg_product_ratings);
			m_resemblance(user1, user2) = c;
			m_resemblance(user2, user1) = c;
		}
		else
		{
			c = m_resemblance(user1, user2);
		}
		return c*c;
	}
	typedef float distance_type;
private:
	const R &m_avg_product_ratings;
	M &m_resemblance;
};

template <class M, class V>
void knn(M &knn_predict, size_t k, 
			const M &users_ratings, const M &user_resemblance_unused, 
			const V &avg_users_rating, const V &avg_product_ratings)
{
	M user_resemblance(user_resemblance_unused.rows(), user_resemblance_unused.cols());
	user_resemblance.zeros();
	KDTree::KDTree<3, itpp::vec, KDTree::_Bracket_accessor<itpp::vec>, 
					kdtree_distance_t<itpp::vec, itpp::mat> > 
				tree(KDTree::_Bracket_accessor<itpp::vec>(), 
						kdtree_distance_t<itpp::vec, itpp::mat>
						(avg_product_ratings, user_resemblance));
	for (size_t i=0; i<users_ratings.rows(); ++i)	//users
	{
		tree.insert(users_ratings.get_row(i));
	}
	
	for (size_t i=0; i<users_ratings.rows(); ++i)	//users
	{
		itpp::mat nearest_neighbours;
		std::vector<itpp::vec> neighbours;
		
		tree.find_within_range(users_ratings.get_row(i), k, 
				std::back_insert_iterator<std::vector<itpp::vec>>(neighbours));
		std::for_each(neighbours.begin(), neighbours.end(), 
					  [&nearest_neighbours](const itpp::vec &v){
						  nearest_neighbours.append_row(v); 
					});
		
		for (size_t j=0; j<users_ratings.cols(); ++j)	//products
		{
			knn_predict(i,j) = grouplens(avg_product_ratings, 
										nearest_neighbours, 
										avg_users_rating, i, j, 
										user_resemblance);
		}
	}
}

template <class T>
void cross_validation_get_sets(const T &triplets, T &validation, T &learning, float validation_rel_size = 0.1)
{
	size_t triplets_count = triplets.size();
	size_t validation_size = triplets_count * validation_rel_size;
	size_t learning_size = triplets_count - validation_size;

	validation.reserve(validation_size);
	learning.reserve(learning_size);

	size_t i = 0;
	while (i<validation_size)
	{
		validation.push_back(triplets[i]);
		++i;
	}
	while (i<triplets_count)
	{
		learning.push_back(triplets[i]);
		++i;
	}
}

template <class M, class T>
void convert_triplets_to_matrix(M &matrix, const T &triplets, 
								const typename T::value_type &max_triplet_values)
{
	matrix.set_size(max_triplet_values.user+1, max_triplet_values.product+1);
	matrix.zeros();

	std::for_each(triplets.begin(), triplets.end(), 
		[&matrix](const T::value_type &x){
			if (x.user >= matrix.rows() || x.product >= matrix.cols())
			{
				std::cout << "convert_triplets_to_matrix: resizing matrix" << std::endl;
				matrix.set_size(x.user+1, x.product+1, true);
			}
			matrix(x.user, x.product) = x.rating;
	});
}

template <class T>
void cross_validation(const T &triplets, 
						const typename T::value_type &max_triplet_values, 
						bool prefer_cached_data)
{
	T validation_triplets;
	T learning_triplets;
	std::cout << "Generating cross-validation data sets...";
	cross_validation_get_sets(triplets, validation_triplets, learning_triplets, 0.1);
	std::cout << "Done." << std::endl;
	
	itpp::mat learning;
	itpp::mat validation;
	learning.zeros();
	validation.zeros();

	std::cout << "Converting triplets to matrices...";
	convert_triplets_to_matrix(learning, learning_triplets, max_triplet_values);
	convert_triplets_to_matrix(validation, validation_triplets, max_triplet_values);
	std::cout << "Done." << std::endl;

	// Average user's and product's ratings
	std::cout << "Average user's and product's ratings...";
	itpp::vec avg_users_rating(learning.rows());
	itpp::vec avg_product_ratings(learning.cols());
	avg_users_rating.zeros();
	avg_product_ratings.zeros();
	avg_ratings(learning, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// Users' resemblance
	std::cout << "Users' resemblance...";
	itpp::mat user_resemblance(learning.rows(), learning.rows());
	user_resemblance.zeros();
	if (prefer_cached_data)
	{
		itpp::it_ifile users_resembl_itpp_file("users_resembl.it");
		std::cout << "Loading cached data...";
		users_resembl_itpp_file >> itpp::Name("users_resembl") >> user_resemblance;
		users_resembl_itpp_file.close();
	}
	else
	{
		std::cout << "Computing...";
		user_resembl(learning, user_resemblance, correlation_coeff_resembl_metric_t<itpp::vec>(avg_product_ratings));
		std::cout << "Caching data...";
		itpp::it_file users_resembl_itpp_file("users_resembl.it");
		users_resembl_itpp_file << itpp::Name("users_resembl") << user_resemblance;
		users_resembl_itpp_file.close();
	}
	std::cout << "Done." << std::endl;
	
	// GroupLens
	std::cout << "GroupLens...";
	itpp::mat grouplens_predict(learning.rows(), learning.cols());
	grouplens_predict.zeros();
	grouplens(grouplens_predict, learning, user_resemblance, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// k-NN
	std::cout << "k-NN...";
	itpp::mat knn_predict(learning.rows(), learning.cols());
	knn_predict.zeros();
	knn(knn_predict, 3, learning, user_resemblance, avg_users_rating, avg_product_ratings);
	std::cout << "Done." << std::endl;
	
	// Validation
	// validation and learning data should be of equal dimensions
	validation.set_size(learning.rows(), learning.cols(), true);

	float grouplens_rmse = rmse(validation, grouplens_predict);
	float knn_rmse = rmse(validation, knn_predict);

	// Output results
	std::cout << "Validation data: \n" << validation << std::endl;
	
	std::cout << "GroupLens: \n" << grouplens_predict << std::endl;
	std::cout << "RMSE: \n" << grouplens_rmse << std::endl;
	
	std::cout << "k-NN: \n" << knn_predict << std::endl;
	std::cout << "RMSE: \n" << knn_rmse << std::endl;
}

int main(int argc, char **argv)
{
	std::string input_filename("");
	std::string output_filename("out.csv");
	size_t input_limit = 0;
	bool load_cached_data = false;

	try
	{
		TCLAP::CmdLine cmd("Recommender", ' ', "0.0");

		TCLAP::ValueArg<std::string> input_filename_arg("i", "input-filename", 
										"Input filename", 
										false, 
										input_filename, 
										"string", 
										cmd);
		
		TCLAP::ValueArg<std::string> output_filename_arg("o", "output-filename", 
										"Output filename", 
										false, 
										output_filename, 
										"string", 
										cmd);

		TCLAP::ValueArg<size_t> input_limit_arg("l", "input-limit", 
										"Input limit (triplets)", 
										false, 
										input_limit, 
										"unsigned integer", 
										cmd);
		
		TCLAP::SwitchArg load_cached_data_arg("c", "load-cached-data", 
										"Load cached auxiliary data", 
										cmd, 
										load_cached_data);
		
		// parse command line
		cmd.parse(argc, argv);

		input_filename = input_filename_arg.getValue();
		output_filename = output_filename_arg.getValue();
		input_limit = input_limit_arg.getValue();
		load_cached_data = load_cached_data_arg.getValue();
	}
	catch(TCLAP::ArgException &excp)
	{
		std::cerr << "TCLAP Error: " << excp.error();
		return 1;
	}
	
	if (setvbuf(stdout, NULL, _IONBF, 0) != 0)	// unbuffered
	//if (setvbuf(stdout, NULL, _IOLBF, 0) != 0)	// line buffering of stdout
	{
		perror("setvbuf");
	}
	
	std::cout << "Reading dataset...";
	std::cin.imbue(std::locale(std::locale(), new csv_locale_facet()));
	std::vector<dataset_triplet_t> triplet_list;
	triplet_list.reserve(input_limit == 0 ? 3000 : input_limit);
	dataset_triplet_t triplet = {0, 0, 0};
	dataset_triplet_t max_triplet_values = {0, 0, 0};
	
	while ((input_limit == 0 || triplet_list.size() < input_limit) 
			&& std::cin >> triplet.user >> triplet.product >> triplet.rating)
	{
		std::cout << "(" << triplet.user << "," << triplet.product << ") -> " << triplet.rating << std::endl;

		max_triplet_values.user = std::max(max_triplet_values.user, triplet.user);
		max_triplet_values.product = std::max(max_triplet_values.product, triplet.product);
		triplet_list.push_back(triplet);
	}
	std::cout << "Done." << std::endl;

	cross_validation(triplet_list, max_triplet_values, load_cached_data);

	return 0;
}
