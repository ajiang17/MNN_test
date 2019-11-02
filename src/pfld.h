#ifndef _FACE_PFLD_H_
#define _FACE_PFLD_H_
#include "opencv2/core.hpp"
#include <vector>

class PFLD {
public:
    PFLD();
    ~PFLD();
    int LoadModel(const char* root_path);
    int ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints);

private:
    class Impl;
    Impl* impl_;
};


#endif