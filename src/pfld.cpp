#include "pfld.h"
#include <iostream>
#include <string>

#include "revertMNNModel.hpp"
#include "Interpreter.hpp"
#include "Backend.hpp"
#include "Tensor.hpp"

#include "opencv2/imgproc.hpp"

class PFLD::Impl {
public:
    Impl() {
        device_ = 0;
        precision_ = 0;
        power_ = 0;
        memory_ = 0;
        
        initialized_ = false;
    }
    ~Impl() {
        landmarker_->releaseModel();
        landmarker_->releaseSession(session_);
    }

    int LoadModel(const char* root_path);
    int ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints);    

    std::shared_ptr<MNN::Interpreter> landmarker_;
    const int inputSize_ = 96;
    
    int device_;
    int precision_;
    int power_;
    int memory_;

    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;
    bool initialized_;
};

PFLD::PFLD() {
    impl_ = new PFLD::Impl();
}

PFLD::~PFLD() {
    if (impl_) {
        delete impl_;
    }
}

int PFLD::Impl::LoadModel(const char* root_path) {
    std::string model_file = std::string(root_path) + "/pfld-lite.mnn";
    landmarker_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    
    MNN::ScheduleConfig config;
    config.numThread = 1;
    config.type      = static_cast<MNNForwardType>(device_);

    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision_;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power_;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory_;
    config.backendConfig = &backendConfig;
    session_ = landmarker_->createSession(config);

    // nhwc to nchw
    std::vector<int> dims{1, inputSize_, inputSize_, 3};
    input_tensor_ = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);

    initialized_ = true;

    return 0;
}

int PFLD::Impl::ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints) {
    std::cout << "start extract keypoints." << std::endl;
    keypoints->clear();
    if (!initialized_) {
        std::cout << "model uninitialed." << std::endl;
        return 10000;
    }
    if (img_face.empty()) {
        std::cout << "input empty." << std::endl;
        return 10001;
    }
    // image prepocess
    cv::Mat face_cpy = img_face.clone();
    int width = face_cpy.cols;
    int height = face_cpy.rows;
    float scale_x = static_cast<float>(width) / inputSize_;
    float scale_y = static_cast<float>(height) / inputSize_;

    cv::Mat face_resized;
    cv::resize(face_cpy, face_resized, cv::Size(inputSize_, inputSize_));
    face_resized.convertTo(face_resized, CV_32FC3);
    face_resized = (face_resized - 123.0f) / 58.0f;

    auto tensor_data = input_tensor_->host<float>();
    auto tensor_size = input_tensor_->size();
    ::memcpy(tensor_data, face_resized.data, tensor_size);

    auto input_tensor = landmarker_->getSessionInput(session_, nullptr);
    input_tensor->copyFromHostTensor(input_tensor_);
    landmarker_->runSession(session_);

    // get output
    std::string output_tensor_name0 = "conv5_fwd";
    MNN::Tensor* tensor_landmarks = landmarker_->getSessionOutput(session_, output_tensor_name0.c_str());
    MNN::Tensor tensor_landmarks_host(tensor_landmarks, tensor_landmarks->getDimensionType());
    tensor_landmarks->copyToHostTensor(&tensor_landmarks_host);

    std::cout << "batch:    " << tensor_landmarks->batch()    << std::endl 
              << "channels: " << tensor_landmarks->channel()  << std::endl
              << "height:   " << tensor_landmarks->height()   << std::endl
              << "width:    " << tensor_landmarks->width()    << std::endl
              << "type:     " << tensor_landmarks->getDimensionType() << std::endl; 

    auto landmarks_dataPtr = tensor_landmarks_host.host<float>();
    int num_of_points = 98;
    for (int i = 0; i < num_of_points; ++i) {
        cv::Point2f curr_pt(landmarks_dataPtr[2 * i + 0] * scale_x,
                            landmarks_dataPtr[2 * i + 1] * scale_y);
        keypoints->push_back(curr_pt);
    }

    std::cout << "end extract keypoints." << std::endl;

    return 0;
}

int PFLD::LoadModel(const char* root_path) {
    return impl_->LoadModel(root_path);
}

int PFLD::ExtractKeypoints(const cv::Mat& img_face, std::vector<cv::Point2f>* keypoints) {
    return impl_->ExtractKeypoints(img_face, keypoints);
}
