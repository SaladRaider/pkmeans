#include <iostream>
#include <boost/program_options.hpp>
#include <pkmeans/pkmeans.h>

using namespace pkmeans;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv) {
    po::options_description desc("pkmeans, a parallized k-means++ application using EMD.\n");

    desc.add_options()
        ("help,?",                                      "produce help message")
        ("clusters,k",      po::value<unsigned int>(),  "number of clusters")
        ("threads,t",       po::value<unsigned int>(),  "community cards for he/o/o8")
        ("in_filename,i",   po::value<std::string>(),   "a hand for evaluation")
        ("out_filename,o",  po::value<std::string>(),   "produces no output");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                  .style(po::command_line_style::unix_style)
                  .options(desc)
                  .run(),
              vm);
    po::notify(vm);

    // check for help
    if (vm.count("help") || argc == 1)
    {
        std::cout << desc << std::endl;
        return 1;
    }

    unsigned int numClusters = vm["clusters"].as<unsigned int>();
    unsigned int numThreads = vm["threads"].as<unsigned int>();
    std::string inFilename = vm["in_filename"].as<std::string>();
    std::string outFilename = vm["out_filename"].as<std::string>();

    PKMeans pkmeans;
    pkmeans.run (numClusters, numThreads, std::string (inFilename),
                 std::string (outFilename));
    return 0;
}
