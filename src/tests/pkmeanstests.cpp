#include <gtest/gtest.h>
#include <pkmeans/pkmeans.h>
#include <string>

using namespace pkmeans;

class PKMeansTests : public ::testing::Test {
protected:
    // You can remove any or all of the following functions if its body
    // is empty.

    PKMeansTests () {
        // You can do set-up work for each test here.
    }

    virtual ~PKMeansTests () {
        // You can do clean-up work that doesn't throw exceptions here.
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

    PKMeans<unsigned int> pkmeans1;
    PKMeans<unsigned long long> pkmeans2;
    PKMeans<double> pkmeans3;
    PKMeans<double> pkmeans4;
};

TEST_F (PKMeansTests, Run) {
    pkmeans1.run (10, 50, 32, "in.txt", "out.txt");
    pkmeans2.run (10, 50, 32, "in.txt", "out.txt");
    pkmeans3.run (10, 50, 32, "in.txt", "out.txt");
    pkmeans4.run (10, 50, 32, "in.txt", "out.txt");
}
