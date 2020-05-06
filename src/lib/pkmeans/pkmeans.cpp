#include <string>
#include <fstream>
#include <unordered_set>
#include "pkmeans.h"

using namespace pkmeans;

void PKMeans::run (int numClusters, int numThreads,
                   std::string inFilename,
                   std::string assignmentsOut,
                   std::string clustersOut) {
    printf ("Running pkmeans with args (k=%d, t=%d, i=%s, a=%s, c=%s)\n",
            numClusters, numThreads, inFilename.c_str (),
            assignmentsOut.c_str (), clustersOut.c_str());

    unsigned int numIterations = 0;
    readDistributions (inFilename);
    initClusters (numClusters);
    assignDistributions ();
    while (!converged) {
        computeNewClusters ();
        assignDistributions ();
        numIterations += 1;
    }
    saveAssignments (assignmentsOut);
    saveClusters (clustersOut);

    printf ("pkmeans finished running with %u iterations.\n", numIterations);
}

void PKMeans::readDistributions (std::string inFilename) {
    std::ifstream infile;
    std::string fileBuffer;
    infile.open (inFilename);
    while (infile) {
        Distribution<double> newDistribution;
        if (!(infile >> newDistribution))
            break;
        distributions.emplace_back (newDistribution);
        prevClusterAssignments.emplace_back (size_t (-1));
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
    std::unordered_set<size_t> takenIdxs;
    size_t randIdx = size_t (rand () % int(distributions.size ()));
    clusterAssignments.clear ();
    for (size_t i = 0; i < k; i++) {
        while (takenIdxs.find (randIdx) != takenIdxs.end ()) {
            randIdx = size_t (rand () % int(distributions.size ()));
        }
        takenIdxs.insert (randIdx);
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
    converged = true;
    clearClusterAssignments ();
    size_t closestClusterIdx;
    for (size_t i = 0; i < distributions.size (); i++) {
        closestClusterIdx = findClosestCluster (i);
        clusterAssignments[closestClusterIdx].emplace_back (i);
        if (closestClusterIdx != prevClusterAssignments[i]) {
            converged = false;
            prevClusterAssignments[i] = closestClusterIdx;
        }
    }
}

void PKMeans::computeClusterMean (size_t idx, Distribution<double> &cluster) {
    cluster = distributions[0];
    cluster.fill (0.0);
    for (size_t i = 0; i < clusterAssignments[idx].size (); i++) {
        cluster += distributions [clusterAssignments [idx][i]];
    }
    cluster /= double(clusterAssignments[idx].size ());
}

void PKMeans::computeNewClusters () {
    Distribution<double> newCluster;
    for (size_t i = 0; i < clusters.size (); i++) {
        computeClusterMean (i, newCluster);
        clusters[i] = newCluster;
    }
}

