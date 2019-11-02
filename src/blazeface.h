#ifndef _FACE_BLAZEFACE_H_
#define _FACE_BLAZEFACE_H_

#include <vector>
#include "opencv2/core.hpp"

struct FaceInfo {
    cv::Rect face_;
    float score_;
};

class BlazeFace {
public:
    BlazeFace();
    ~BlazeFace();

    int LoadModel(const char* root_path);
    int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

private:
    class Impl;
    Impl* impl_;
};

#endif // !_FACE_BLAZEFACE_H_

