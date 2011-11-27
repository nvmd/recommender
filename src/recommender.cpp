
#include <cstddef>
#include <iostream>
#include <vector>
#include <string>

#include <tclap/CmdLine.h>

#include "dataset_io.hpp"
#include "cross_validation.hpp"

int main(int argc, char **argv)
{
	std::string input_filename("");
	std::string output_filename("out.csv");
	size_t skip_lines = 0;
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
	
		TCLAP::ValueArg<size_t> skip_lines_arg("s", "skip-lines", 
										"Skip lines at the beginning of the input file", 
										false, 
										skip_lines, 
										"unsigned integer", 
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
		skip_lines = skip_lines_arg.getValue();
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
	std::vector<dataset_triplet_t> triplet_list;
	const size_t triplet_list_reserve = 3000;
	dataset_triplet_t max_triplet_values = {0, 0, 0};
	triplet_list.reserve(input_limit == 0 ? triplet_list_reserve : input_limit);
	if (input_filename.empty() || input_filename == "-")
	{
		read_dataset(std::cin, triplet_list, max_triplet_values, 
					 input_limit, skip_lines);
	}
	else
	{
		std::ifstream input_file(input_filename);
		if (input_file.is_open())
		{
			read_dataset(input_file, triplet_list, max_triplet_values, 
						 input_limit, skip_lines);
			input_file.close();
		}
		else
		{
			std::cout << "Can't open file: \"" 
					  << input_filename << "\"" << std::endl;
		}
	}
	std::cout << "Done." << std::endl;

	cross_validation(triplet_list, max_triplet_values, load_cached_data);

	return 0;
}
