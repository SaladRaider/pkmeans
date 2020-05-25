#include <pkmeans/pkmeans.h>
#include <pkmeans/pkmeans_lowmem.h>
#include <time.h>

#include <boost/program_options.hpp>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace pkmeans;
using namespace std;
namespace po = boost::program_options;

int main(int argc, char **argv) {
  // Make I/O faster
  ios_base::sync_with_stdio(false);
  std::cin.tie(NULL);

  po::options_description desc(
      "pkmeans, a parallized k-means++ application using EMD.\n");

  desc.add_options ()
    ("help,?",              "produce help message")
    ("clusters,k",          po::value<unsigned int> ()->default_value (8),                      "number of clusters")
    ("threads,t",           po::value<unsigned int> ()->default_value (1),                      "number of threads")
    ("in_filename,i",       po::value<std::string> ()->default_value ("in.txt"),                "input filename for distributions")
    ("assignments_out,a",   po::value<std::string> ()->default_value ("assignments_out.txt"),   "cluster assignments to distributions output filename")
    ("clusters_out,c",      po::value<std::string> ()->default_value ("clusters_out.txt"),      "clusters output filename")
    ("confidence_prob,p",   po::value<float> ()->default_value       (0.1f),                    "confidence probability of the stopping criteria upper bound for random restarts")
    ("missing_mass,m",      po::value<float> ()->default_value       (0.1f),                    "maximum acceptable missing mass for random restarts")
    ("seed,s",              po::value<size_t> ()->default_value      (size_t(-1)),              "starting random seed")
    ("backup_every,b",      po::value<size_t> ()->default_value      (0),                       "backup every b itterations")
    ("euclidean,e",         "use euclidean L2 distance instead of earth mover's distance")
    ("low_mem,l",           "use low memory version")
    ("quiet,q",             "reduce logs in console");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .style(po::command_line_style::unix_style)
                .options(desc)
                .run(),
            vm);
  po::notify(vm);

  // check for help
  if (vm.count("help") || argc == 1) {
    std::cout << desc << std::endl;
    return 1;
  }

  const auto numClusters = vm["clusters"].as<unsigned int>();
  const auto numThreads = vm["threads"].as<unsigned int>();
  const auto confidenceProb = vm["confidence_prob"].as<float>();
  const auto missingMass = vm["missing_mass"].as<float>();
  const auto seed = vm["seed"].as<size_t>();
  const auto useTimeSeed = size_t(-1) == seed;
  const auto backupEvery = vm["backup_every"].as<size_t>();
  const auto inFilename = vm["in_filename"].as<std::string>();
  const auto assignmentsOut = vm["assignments_out"].as<std::string>();
  const auto clustersOut = vm["clusters_out"].as<std::string>();
  const auto lowMem = vm.count("low_mem") > 0;
  const auto euclidean = vm.count("euclidean") > 0;
  const auto quiet = vm.count("quiet") > 0;

  if (lowMem) {
    auto pkmeans = std::make_unique<PKMeans<std::uint8_t>>();
    pkmeans->run(numClusters, numThreads, confidenceProb, missingMass, seed,
                 backupEvery, useTimeSeed, inFilename, assignmentsOut,
                 clustersOut, euclidean, quiet);
  } else {
    auto pkmeans = std::make_unique<PKMeans<float>>();
    pkmeans->run(numClusters, numThreads, confidenceProb, missingMass, seed,
                 backupEvery, useTimeSeed, inFilename, assignmentsOut,
                 clustersOut, euclidean, quiet);
  }
  return 0;
}
