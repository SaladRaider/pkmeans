#include <iostream>
#include <string>
#include "pkmeans.h"

int main() {
    PKMeans pkmeans;
    pkmeans.run(10, 32, "distributions.txt", "clusters.txt");
    return 0;
}
