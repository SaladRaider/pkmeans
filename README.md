# Parallelized k-means++ w/ EMD
Copyright (c) 2020 Peter Kang Veerman

**pkmean** is a fast parallelized k-mean++  clustering application using the triangle inequality  and one dimensional Earth Mover's Distance. It is capable of clustering billions of distributions.

**pkmean** is licensed under the MIT license.

## Features

* Distance function is one dimensional Earth Mover's Distance.
* Uses triangle inequality \[1\]
* Parallelized \[2\]
* Implements k-means++ \[3\] for starting cluster locations.
* Can process large datasets

## Usage
##### Example
The following is an example of clustering points in the file `distributions.txt` with 10 clusters and saving the cluster distributions to the file `clusters.txt` using 32 threads.
```shell
$ ./pkmean -k 10 -t 32 -i distributions.txt -o clusters.txt
```

##### distributions.txt
This is an example file of distributions with 10 buckets. The values for each bucket must be an integer. **pkmeans** will normalize the distributions when run. The integers are space-seperated.
```
1 0 4 2 5 6 2 3 0 2
0 3 0 2 0 9 0 0 0 3
2 8 2 3 0 0 3 2 4 0
...
```


## Installation
### Build From Source
#### Requirements

```
cmake
ccmake
pthreads
```

#### Building

```shell
$ cd build
$ ccmake ../

$ make
$ sudo make install
```

## References
[\[1\]](https://www.aaai.org/Papers/ICML/2003/ICML03-022.pdf) Charles Elkan. Using the triangle inequality to accelerate k-means.
*International Conference on Machine Learning*, 2003.

[\[2\]](http://vldb.org/pvldb/vol5/p622_bahmanbahmani_vldb2012.pdf) Bahmani, B. Moseley, A. Vattani, R. Kumar, and S. Vassilvitskii. 2012b. Scalable K-Means++. *PVLDB* 5,7 (2012), 622–633

[\[3\]](http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf) D. Arthur and S. Vassilvitskii, “K-Means++: The advantages of
careful seeding,” in *Proc. Symp. Discrete Algorithms*, 2007,
pp. 1027–1035.
