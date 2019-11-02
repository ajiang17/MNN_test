#include "blazeface.h"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

int main(int argc, char* argv[]) {
    const char* image_path = "../data/images/image_04.jpg";
    cv::Mat img_src = cv::imread(image_path);

    const char* root_path = "../data/models";
    BlazeFace* blazeface = new BlazeFace();
    blazeface->LoadModel(root_path);

    std::vector<FaceInfo> faces;
    blazeface->Detect(img_src, &faces);
    int face_num = static_cast<int>(faces.size());
    for (int i = 0; i < face_num; ++i) {
        cv::rectangle(img_src, faces[i].face_, cv::Scalar(255, 0, 255), 2);
    }

    const char* result_path = "../data/images/detect_result.jpg";
    cv::imwrite(result_path, img_src);

    return 0;
}