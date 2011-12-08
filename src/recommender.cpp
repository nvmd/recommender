
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>

#include <tclap/CmdLine.h>

#include "dataset_io.hpp"
#include "cross_validation.hpp"

int main(int argc, char **argv)
{
	bool do_cross_validation = true;
	std::string recommendation_request_filename("");
	
	std::string input_filename("");
	size_t skip_lines = 0;
	size_t input_limit = 0;
	
	size_t recom_skip_lines = 0;
	size_t recom_input_limit = 0;
	std::string recom_output_filename("out.csv");
	
	bool load_cached_data = false;
	size_t output_verbosity = 0;

	try
	{
		TCLAP::CmdLine cmd("Recommender", ' ', "0.0");
		
		// Modes of operation: cross-validation, recommendation
		TCLAP::SwitchArg cross_validation_arg("v", "cross-validation", 
										"Do cross validation", 
										cmd, 
										do_cross_validation);
		
		TCLAP::ValueArg<std::string> recommendation_arg("r", "recommendation", 
										"Recommendation request filename", 
										false, 
										recommendation_request_filename, 
										"string", 
										cmd);
		
		// Dataset
		TCLAP::ValueArg<std::string> input_filename_arg("d", "dataset-filename", 
										"Dataset filename", 
										false, 
										input_filename, 
										"string", 
										cmd);
		
		TCLAP::ValueArg<size_t> skip_lines_arg("s", "dataset-skip-lines", 
										"Skip lines at the beginning of the input file", 
										false, 
										skip_lines, 
										"unsigned integer", 
										cmd);
		
		TCLAP::ValueArg<size_t> input_limit_arg("l", "dataset-input-limit", 
										"Input limit (triplets)", 
										false, 
										input_limit, 
										"unsigned integer", 
										cmd);
		
		// Recommendation request parameters
		TCLAP::ValueArg<size_t> recom_skip_lines_arg("k", "recom-skip-lines", 
										"Skip lines at the beginning of the input file", 
										false, 
										recom_skip_lines, 
										"unsigned integer", 
										cmd);
		
		TCLAP::ValueArg<size_t> recom_input_limit_arg("m", "recom-input-limit", 
										"Input limit (triplets)", 
										false, 
										recom_input_limit, 
										"unsigned integer", 
										cmd);
		
		TCLAP::ValueArg<std::string> output_filename_arg("o", "recom-output-filename", 
										"Output filename", 
										false, 
										recom_output_filename, 
										"string", 
										cmd);
		
		TCLAP::ValueArg<size_t> verbosity_arg("z", "verbosity",
										"Output verbosity",
										false,
										output_verbosity,
										"unsigned integer",
										cmd);
		
		TCLAP::SwitchArg load_cached_data_arg("c", "load-cached-data", 
										"Load cached auxiliary data", 
										cmd, 
										load_cached_data);
		
		
		// Parse command line
		cmd.parse(argc, argv);

		// Read arguments' values
		do_cross_validation = cross_validation_arg.getValue();
		recommendation_request_filename = recommendation_arg.getValue();
		
		input_filename = input_filename_arg.getValue();
		skip_lines = skip_lines_arg.getValue();
		input_limit = input_limit_arg.getValue();
		
		recom_skip_lines = recom_skip_lines_arg.getValue();
		recom_input_limit = recom_input_limit_arg.getValue();
		recom_output_filename = output_filename_arg.getValue();
		
		load_cached_data = load_cached_data_arg.getValue();
		output_verbosity = verbosity_arg.getValue();
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
	
	if (output_verbosity >= 1)
	{
		std::cout << "Reading dataset...";
	}
	std::vector<dataset_triplet_t> triplet_list;
	const size_t triplet_list_reserve = 3000;
	dataset_triplet_t max_triplet_values = {0, 0, 0};
	triplet_list.reserve(input_limit == 0 ? triplet_list_reserve : input_limit);
	if (input_filename.empty() || input_filename == "-")
	{
		read_dataset(std::cin, triplet_list, max_triplet_values, 
					 input_limit, skip_lines, output_verbosity);
	}
	else
	{
		std::ifstream input_file(input_filename);
		if (input_file.is_open())
		{
			read_dataset(input_file, triplet_list, max_triplet_values, 
						 input_limit, skip_lines, output_verbosity);
			input_file.close();
		}
		else
		{
			std::cout << "Can't open file: \"" 
					  << input_filename << "\"" << std::endl;
		}
	}
	if (output_verbosity >= 1)
	{
		std::cout << "Done." << std::endl;
	}

	if (do_cross_validation)
	{
		cross_validation(triplet_list, max_triplet_values, 
						 load_cached_data, output_verbosity);
	}
	
	if (!recommendation_request_filename.empty())
	{
		std::vector<dataset_triplet_t> recom_triplet_list;
		const size_t recom_triplet_list_reserve = 3000;
		dataset_triplet_t recom_max_triplet_values = {0, 0, 0};
		recom_triplet_list.reserve(input_limit == 0 ? recom_triplet_list_reserve : recom_input_limit);
		
		std::ifstream input_file(recommendation_request_filename);
		if (input_file.is_open())
		{
			read_dataset(input_file, recom_triplet_list,recom_max_triplet_values, 
						 recom_input_limit, recom_skip_lines, output_verbosity);
			input_file.close();
		}
		else
		{
			std::cout << "Can't open file: \"" 
					  << input_filename << "\"" << std::endl;
		}
		
		// launch recommender_t
		std::cout << "Recommendations will be here someday." << std::endl;
	}

	return 0;
}
