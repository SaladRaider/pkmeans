#include "progress_bar.h"

#include <sys/ioctl.h>
#include <unistd.h>
#include <cstring>

#include <iomanip>
#include <iostream>

using namespace pkmeans;

template <size_t N>
ProgressBar<N>::ProgressBar()
    : progress{0},
      prevProgress{0},
      lastShownTime{std::chrono::steady_clock::now()},
      numShows{0},
      elapsedSum{0} {
  memset(progressBar, '\0', sizeof(char) * (progressSize + 1));
}

template <size_t N>
void ProgressBar<N>::show(float progressPercent) {
  progress = progressSize * progressPercent;
  if (prevProgress != progress) {
    auto newTime = std::chrono::steady_clock::now();
    numShows *= timeDecay;
    elapsedSum *= timeDecay;
    numShows += 1;
    elapsedSum += std::chrono::duration_cast<std::chrono::milliseconds>(
                      newTime - lastShownTime)
                      .count() *
                  (progressSize - progress) / 1000.f;
    auto minutesLeft = static_cast<int>(elapsedSum / numShows / 60) % 60;
    auto hoursLeft = static_cast<int>(elapsedSum / numShows / 60 / 60);
    auto secondsLeft = static_cast<int>(elapsedSum / numShows) % 60;
    lastShownTime = newTime;
    prevProgress = progress;
    memset(progressBar, ' ', sizeof(char) * progressSize);
    memset(progressBar, '=', sizeof(char) * progress);
    if (progress < progressSize)
      memset(progressBar + progress, '>', sizeof(char) * 1);
    std::cout << std::left << "\r\033[1;32m[" << progressBar << "] "
              << std::setw(6) << std::fixed << std::setprecision(2)
              << (progressPercent * 100) << "% ";
    if (hoursLeft > 0)
      std::cout << std::right << std::setw(3) << hoursLeft << "h ";
    if (hoursLeft > 0 || minutesLeft > 0)
      std::cout << std::right << std::setw(2) << minutesLeft << "m ";
    std::cout << std::right << std::setw(2) << secondsLeft << "s remaining";
    std::cout << "\033[0m";
    std::cout << std::flush;
  }
}

namespace pkmeans {
template class ProgressBar<50>;
}  // namespace pkmeans
