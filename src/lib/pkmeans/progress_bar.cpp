#include "progress_bar.h"

#include <sys/ioctl.h>
#include <tgmath.h>
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
      elapsedSum{0},
      elapsedMs{0},
      lastProgressPercent{0} {
  memset(progressBar, '\0', sizeof(char) * (progressSize + 1));
}

template <size_t N>
void ProgressBar<N>::show(float progressPercent) {
  progress = fmin(progressSize * progressPercent, progressSize);
  currTime = std::chrono::steady_clock::now();
  elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                  currTime - lastShownTime)
                  .count();
  if (lastProgressPercent != progressPercent &&
      (progress != prevProgress || elapsedMs > showTimeMs)) {
    if (progress != prevProgress) {
      numShows *= timeDecay;
      elapsedSum *= timeDecay;
    }
    numShows += 1.f;
    elapsedSum += elapsedMs / (progressPercent - lastProgressPercent) / 1000.f;
    auto minutesLeft =
        static_cast<int>(elapsedSum * (1.f - progressPercent) / numShows / 60) %
        60;
    auto hoursLeft = static_cast<int>(elapsedSum * (1.f - progressPercent) /
                                      numShows / 60 / 60);
    auto secondsLeft =
        static_cast<int>(elapsedSum * (1.f - progressPercent) / numShows) % 60;
    lastShownTime = currTime;
    lastProgressPercent = progressPercent;
    prevProgress = progress;
    elapsedMs = 0;
    memset(progressBar, ' ', sizeof(char) * progressSize);
    memset(progressBar, '=', sizeof(char) * progress);
    if (progress < progressSize)
      memset(progressBar + progress, '>', sizeof(char) * 1);
    std::cout << std::left << "\r\033[1;32m[" << progressBar << "] "
              << std::right << std::setw(5) << std::fixed
              << std::setprecision(2) << (progressPercent * 100) << "% ";
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
