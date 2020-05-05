#include <gtest/gtest.h>
#include <pkmeans/distribution.h>
#include <string>
#include <array>
#include <time.h>
#include <fstream>

using namespace pkmeans;

class DistributionTests : public ::testing::Test {
protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    DistributionTests () {
        srand (time (NULL));

        // You can do set-up work for each test here.
        // Initialize random distributions
        for (size_t j = 0; j < 4; j++)
        for (size_t i = 0; i < 50; i++) {
            dummyBuckets[j][i] = rand () % 100;
        }

        // Create test distribution file
        std::ofstream f;
        f.open("test_distribution.txt");
        for (size_t j = 0; j < 4; j++)
        for (size_t i = 0; i < dummyBuckets.size (); i++) {
            for (size_t j = 0; j < 49; j++) {
                f << dummyBuckets[i][j] << ' ';
            }
            f << dummyBuckets[i][49] << '\n';
        }
        f.close ();

        // Stream test distribution file
        std::ifstream infile;
        std::string fileBuffer;
        infile.open ("test_distribution.txt");
        for (size_t j = 0; j < 4; j++) {
            infile >> distributions[j];
        }
        infile.close ();
    }

    virtual ~DistributionTests () {
        // You can do clean-up work that doesn't throw exceptions here.
        std::remove ("test_dummyBuckets.txt");
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

    static constexpr size_t numDistributions = 4;
    std::array<Distribution<unsigned int>, numDistributions> distributions;
    std::array<std::array<int, 50>, numDistributions> dummyBuckets;
};

TEST_F (DistributionTests, SizeTest) {
    for (size_t i = 0; i < distributions.size (); i++) {
        EXPECT_EQ (50, distributions[i].size ());
    }
}

TEST_F (DistributionTests, FillTest) {
    int randVal = rand () % 100;
    for (size_t i = 0; i < distributions.size (); i++) {
        distributions[i].fill (randVal);
        for (size_t j = 0; j < distributions[i].size (); j++) {
            EXPECT_EQ (randVal, distributions[i][j]);
        }
    }
}
