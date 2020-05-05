#include "distribution.h"

using namespace pkmeans;

template <class T>
Distribution<T>::Distribution () {}

template <class T>
const size_t Distribution<T>::size() {
    return buckets.size ();
}

template <class T>
void Distribution<T>::fill (T value) {
    buckets.fill (value);
}

template <class T>
T& Distribution<T>::operator[] (size_t idx) {
    return buckets[idx];
}

template <class T>
Distribution& Distribution<T>::operator= (const Distribution &other) {
    if (this != &other) {
        std::copy (other.buckets.begin (), other.buckets.end (),
                   std::back_inserter (buckets));
    }
    return *this;
}

template <class T>
Distribution& Distribution<T>::operator+= (const Distribution &other) {
    for (size_t i = 0; i < size (); i++) {
        buckets[i] += other.buckets[i];
    }
    return *this;
}

template <class T>
Distribution& Distribution<T>::operator-= (const Distribution &other) {
    for (size_t i = 0; i < size (); i++) {
        buckets[i] -= other.buckets[i];
    }
    return *this;
}

template <class T>
friend Distribution Distribution<T>::operator+ (Distribution lhs, const Distribution &rhs) {
    lhs += rhs;
    return lhs;
}

template <class T>
friend Distribution Distribution<T>::operator- (Distribution lhs, const Distribution &rhs) {
    lhs -= rhs;
    return lhs;
}

template <class T>
unsigned int Distribution<T>::emd(const Distribution &d1, const Distribution &d2) {
    return 0;
}
