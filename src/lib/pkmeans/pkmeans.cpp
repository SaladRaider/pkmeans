#include <string>
#include <fstream>
#include <unordered_set>
#include "pkmeans.h"

using namespace pkmeans;

void PKMeans::run (int numClusters, int numThreads, std::string inFilename, std::string outFilename) {
    printf ("Running pkmeans with args (k=%d, t=%d, i=%s, o=%s)\n",
            numClusters, numThreads, inFilename.c_str (),
            outFilename.c_str ());

    readDistributions (inFilename);
    initClusters (numClusters);
    while (!converged) {
        assignDistributions ();
        computeNewClusters ();
    }
    saveClusters (outFilename);

    printf ("pkmeans finished running.");
}

void PKMeans::readDistributions (std::string inFilename) {
    std::ifstream infile;
    std::string fileBuffer;
    infile.open (inFilename);
    Distribution<double> newDistribution;
    while (infile) {
        if (!(infile >> newDistribution))
            break;
        distributions.emplace_back (newDistribution);
    }
    infile.close ();
}

void PKMeans::saveClusters (std::string outFilename) {
    std::ofstream f;
    f.open (outFilename);
    for (size_t i = 0; i < clusters.size (); i++) {
        f << clusters[i] << '\n';
    }
    f.close ();
}

void PKMeans::saveAssignments (std::string outFilename) {
    std::ofstream f;
    f.open (outFilename);
    for (size_t i = 0; i < clusterAssignments.size (); i++) {
        for (size_t j = 0; j < clusterAssignments[i].size (); j++) {
            f << i << ' ' << clusterAssignments[i][j] << '\n';
        }
    }
    f.close ();
}

void PKMeans::initClusters (int numClusters) {
    size_t k = size_t (numClusters);
    std::unordered_set<size_t> takendIdxs;
    size_t randIdx = size_t (rand () % int(distributions.size ()));
    for (size_t i = 0; i < k; i++) {
        while (takendIdxs.find (randIdx) != takendIdxs.end ()) {
            randIdx = size_t (rand () % int(distributions.size ()));
        }
        takendIdxs.insert (randIdx);
        clusters.emplace_back (distributions[randIdx]);
        clusterAssignments.emplace_back ();
    }
}

void PKMeans::clearClusterAssignments () {
    for (size_t i = 0; i < clusterAssignments.size (); i++) {
        clusterAssignments[i].clear ();
    }
}

size_t PKMeans::findClosestCluster (size_t distributionIdx) {
    size_t closestIdx = 0;
    double dist = Distribution<double>::emd (distributions[distributionIdx],
                                             clusters[0]);
    double newDist;
    for (size_t i = 1; i < clusters.size (); i++) {
        newDist = Distribution<double>::emd (distributions[distributionIdx],
                                              clusters[i]);
        if (newDist < dist) {
            closestIdx = i;
            dist = newDist;
        }
    }
    return closestIdx;
}

void PKMeans::assignDistributions () {
    clearClusterAssignments ();
    size_t closestClusterIdx;
    for (size_t i = 0; i < distributions.size (); i++) {
        clusterAssignments[findClosestCluster (i)].emplace_back (i);
    }
}

void PKMeans::computeClusterMean (size_t idx, Distribution<double> &cluster) {
    cluster.fill (0.0);
    for (size_t i = 0; i < clusterAssignments[idx].size (); i++) {
        cluster += distributions [clusterAssignments [idx][i]];
    }
    cluster /= double(clusterAssignments[idx].size ());
}

void PKMeans::computeNewClusters () {
    Distribution<double> newCluster;
    converged = true;
    for (size_t i = 0; i < clusters.size (); i++) {
        computeClusterMean (i, newCluster);
        if (clusters[i] != newCluster) {
            clusters[i] = newCluster;
            converged = false;
        }
    }
}

