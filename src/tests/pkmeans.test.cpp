#include <gtest/gtest.h>
#include <pkmeans/pkmeans.h>
#include <pkmeans/distribution.h>
#include <iostream>
#include <string>
#include <array>
#include <fstream>

namespace pkmeans {
class PKMeansTests : public ::testing::Test {
protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    PKMeansTests () {
        srand (time (NULL));

        // Initialize random distributions
        for (size_t j = 0; j < dummyBuckets.size (); j++)
        for (size_t i = 0; i < dummyBuckets[j].size (); i++) {
            dummyBuckets[j][i] = rand () % 100;
        }

        // Create test distribution file
        std::ofstream f;
        f.open ("test_distribution.txt");
        for (size_t i = 0; i < dummyBuckets.size (); i++) {
            for (size_t j = 0; j < 49; j++) {
                f << dummyBuckets[i][j] << ' ';
            }
            f << dummyBuckets[i][49] << '\n';
        }
        f.close ();

        pkmeans.readDistributions ("test_distribution.txt");
    }

    virtual ~PKMeansTests () {
        // You can do clean-up work that doesn't throw exceptions here.
        if ((std::remove ("test_distribution.txt")) != 0)
            printf ("Could not remove test_distribution.txt\n");
    }

    // If the constructor and destructor are not enough for setting up
    // and cleaning up each test, you can define the following methods:
    virtual void SetUp () {
        // Code here will be called immediately after the constructor (right
        // before each test).
    }

    virtual void TearDown () {
        // Code here will be called immediately after each test (right
        // before the destructor).
    }

    PKMeans pkmeans;
    std::array<std::array<double, 50>, 4> dummyBuckets;
};

TEST_F (PKMeansTests, ReadDistributions) {
    ASSERT_EQ(4, pkmeans.distributions.size ());

    for (size_t i = 0; i < dummyBuckets.size (); i++)
    for (size_t j = 0; j < dummyBuckets[i].size (); j++) {
        EXPECT_DOUBLE_EQ (dummyBuckets[i][j],
                          pkmeans.distributions[i][j])
            << "Distribution " << i << " different at bucket " << j << ".\n";
    }
}

TEST_F (PKMeansTests, InitClusters) {
    pkmeans.initClusters (2);

    ASSERT_EQ(2, pkmeans.clusters.size ());
    for (size_t i = 0; i < pkmeans.clusters.size (); i++) {
        ASSERT_EQ(dummyBuckets[0].size (), pkmeans.clusters[i].size ());
    }
}

TEST_F (PKMeansTests, FindClosestCluster) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    EXPECT_EQ (0, pkmeans.findClosestCluster (0))
        << "d(dist0,c0)="
        << Distribution<double>::emd (pkmeans.distributions[0],
                                      pkmeans.clusters[0])
        << "; d(dist0,c1)="
        << Distribution<double>::emd (pkmeans.distributions[0],
                                      pkmeans.clusters[1])
        << '\n';
    EXPECT_EQ (0, pkmeans.findClosestCluster (1))
        << "d(dist1,c0)="
        << Distribution<double>::emd (pkmeans.distributions[1],
                                      pkmeans.clusters[0])
        << "; d(dist1,c1)="
        << Distribution<double>::emd (pkmeans.distributions[1],
                                      pkmeans.clusters[1])
        << '\n';
    EXPECT_EQ (1, pkmeans.findClosestCluster (2))
        << "d(dist2,c0)="
        << Distribution<double>::emd (pkmeans.distributions[2],
                                      pkmeans.clusters[0])
        << "; d(dist2,c1)="
        << Distribution<double>::emd (pkmeans.distributions[2],
                                      pkmeans.clusters[1])
        << '\n';
    EXPECT_EQ (1, pkmeans.findClosestCluster (3))
        << "d(dist3,c0)="
        << Distribution<double>::emd (pkmeans.distributions[3],
                                      pkmeans.clusters[0])
        << "; d(dist3,c1)="
        << Distribution<double>::emd (pkmeans.distributions[3],
                                      pkmeans.clusters[1])
        << '\n';
}

TEST_F (PKMeansTests, AssignDistributions) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.assignDistributions ();

    ASSERT_EQ(2, pkmeans.clusterAssignments.size ());
    ASSERT_EQ(2, pkmeans.clusterAssignments[0].size ());
    ASSERT_EQ(2, pkmeans.clusterAssignments[1].size ());

    EXPECT_EQ(0, pkmeans.clusterAssignments[0][0]);
    EXPECT_EQ(1, pkmeans.clusterAssignments[0][1]);
    EXPECT_EQ(2, pkmeans.clusterAssignments[1][0]);
    EXPECT_EQ(3, pkmeans.clusterAssignments[1][1]);
}

TEST_F (PKMeansTests, ClearClusterAssignments) {
    pkmeans.initClusters (2);
    pkmeans.assignDistributions ();

    ASSERT_GT(pkmeans.clusterAssignments.size (), 0.0);
    for (size_t i = 0; i < pkmeans.clusterAssignments.size (); i++) {
        ASSERT_GT(pkmeans.clusterAssignments[i].size (), 0.0)
            << "cluster " << i << " has 0 assigned distributions.\n";
    }

    pkmeans.clearClusterAssignments ();

    ASSERT_GT(pkmeans.clusterAssignments.size (), 0.0);
    for (size_t i = 0; i < pkmeans.clusterAssignments.size (); i++) {
        ASSERT_EQ(pkmeans.clusterAssignments[i].size (), 0.0);
    }
}

TEST_F (PKMeansTests, ComputeClusterMean) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.assignDistributions ();

    pkmeans.computeClusterMean (0);
    EXPECT_TRUE (
        (pkmeans.distributions[0] + pkmeans.distributions[1]) / 2.0
            == pkmeans.clusters[0]);
    pkmeans.computeClusterMean (1);
    EXPECT_TRUE (
        (pkmeans.distributions[2] + pkmeans.distributions[3]) / 2.0
                == pkmeans.clusters[1]);
}

TEST_F (PKMeansTests, ComputeNewClusters) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.assignDistributions ();
    pkmeans.computeNewClusters ();

    EXPECT_TRUE (
        (pkmeans.distributions[0] + pkmeans.distributions[1]) / 2.0
            == pkmeans.clusters[0]);
    EXPECT_TRUE (
        (pkmeans.distributions[2] + pkmeans.distributions[3]) / 2.0
                == pkmeans.clusters[1]);
}

TEST_F (PKMeansTests, CalcObjFn) {
    for (size_t i = 0; i < pkmeans.distributions.size (); i++) {
        pkmeans.distributions[i].fill (0.0);
    }
    pkmeans.distributions[0][0] = 1.0;
    pkmeans.distributions[0][49] = 1.0;
    pkmeans.distributions[1][1] = 1.0;
    pkmeans.distributions[1][48] = 1.0;
    pkmeans.distributions[2][24] = 1.0;
    pkmeans.distributions[2][25] = 1.0;
    pkmeans.distributions[3][20] = 1.0;
    pkmeans.distributions[3][30] = 1.0;

    pkmeans.initClusters (2);
    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][0] = 1.0;
    pkmeans.clusters[0][49] = 1.0;
    pkmeans.clusters[1][23] = 1.0;
    pkmeans.clusters[1][22] = 1.0;

    pkmeans.assignDistributions ();

    EXPECT_DOUBLE_EQ (15.0, pkmeans.calcObjFn ());

    pkmeans.clusters[0].fill (0.0);
    pkmeans.clusters[1].fill (0.0);
    pkmeans.clusters[0][2] = 1.0;
    pkmeans.clusters[0][47] = 1.0;
    pkmeans.clusters[1][24] = 1.0;
    pkmeans.clusters[1][21] = 1.0;

    pkmeans.assignDistributions ();

    EXPECT_DOUBLE_EQ (17.0, pkmeans.calcObjFn ());
}

}
