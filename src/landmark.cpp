#include <vector>
#include "pfld.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main(int argc, char* argv[]) {
    const char* image_path = "../data/images/face.jpg";
    cv::Mat img_face = cv::imread(image_path);

    const char* root_path = "../data/models";
    PFLD* pfld = new PFLD();
    pfld->LoadModel(root_path);
    std::vector<cv::Point2f> keypoints;
    pfld->ExtractKeypoints(img_face, &keypoints);

    int num_keypoints = static_cast<int>(keypoints.size());
    for (int i = 0; i < num_keypoints; ++i) {
        cv::circle(img_face, keypoints[i], 2, cv::Scalar(0, 255, 255), -1);
    }

    const char* img_result = "../data/images/result.jpg";
    cv::imwrite(img_result, img_face);

    return 0;
}