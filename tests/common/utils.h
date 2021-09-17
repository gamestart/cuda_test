/******************************************************************************
 * FILENAME:      utils.h
 *
 * AUTHORS:       Yutong Huang
 *
 * LAST MODIFIED: Sat 08 May 2021 04:46:36 PM CST
 *
 * CONTACT:       yutong.huang@smartmore.com
 ******************************************************************************/

#ifndef __CUDAOP_TEST_COMMON_UTILS_H__
#define __CUDAOP_TEST_COMMON_UTILS_H__

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace smartmore {
static inline std::string GetFileDirectory(const char *file_path) {
    std::string path = std::string(file_path);
    return std::string(path.begin(), path.begin() + path.rfind('/'));
}

inline float CVMatMaxDiff(cv::Mat actual, cv::Mat expect) {
    if (actual.cols != expect.cols) throw std::runtime_error("actual.cols!=expect.cols");
    if (actual.rows != expect.rows) throw std::runtime_error("actual.rows != expect.rows");
    if (actual.channels() != expect.channels()) throw std::runtime_error("actual.channels() != expect.channels()");

    float max_diff = 0;
    for (int i = 0; i < actual.rows; i++) {
        for (int j = 0; j < actual.cols; j++) {
            for (int k = 0; k < expect.channels(); k++) {
                float diff = fabs(actual.at<cv::Vec3f>(i, j)[k] - expect.at<cv::Vec3f>(i, j)[k]);
                max_diff = diff > max_diff ? diff : max_diff;
            }
        }
    }
    return max_diff;
}

inline void RandomFloatVector(std::vector<float> &vec_f) {
    srand((unsigned int)(time(NULL)));
    std::for_each(vec_f.begin(), vec_f.end(), [](float &f) { f = (rand() % 255) / 255.f; });
}

inline void RandomInt8Vector(std::vector<unsigned char> &vec_uc) {
    srand((unsigned int)(time(NULL)));
    std::for_each(vec_uc.begin(), vec_uc.end(), [](unsigned char &uc) { uc = rand() % 255; });
}

class Clock {
   public:
    Clock() = delete;

    Clock(std::string event) : _event(event) { _start_time = std::chrono::high_resolution_clock::now(); }

    ~Clock() { std::cout << _event << formatTime() << std::endl; };

    double DurationMs() {
        using namespace std::chrono;
        typedef duration<double, std::ratio<1, 1000>> milliSecond;
        milliSecond duration_ms = duration_cast<milliSecond>(high_resolution_clock::now() - _start_time);
        return duration_ms.count();
    };

   private:
    std::string formatTime() {
        auto ms = DurationMs();
        if (ms < 1.0) {
            return std::to_string(ms * 1000.0) + "μs";
        }
        if (ms > 1000.0) {
            return std::to_string(ms / 1000.0) + "s";
        }
        return std::to_string(ms) + "ms";
    }

    std::string _event;
    std::chrono::high_resolution_clock::time_point _start_time;
};

}  // namespace smartmore

#endif  // __UTILS_H__