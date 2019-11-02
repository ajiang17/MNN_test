#include "blazeface.h"
#include <iostream>
#include <string>
#include <algorithm>

#include "revertMNNModel.hpp"
#include "Interpreter.hpp"
#include "Backend.hpp"
#include "Tensor.hpp"

#include "opencv2/imgproc.hpp"

float InterRectArea(const cv::Rect & a, const cv::Rect & b) {
	cv::Point left_top = cv::Point(MAX(a.x, b.x), MAX(a.y, b.y));
	cv::Point right_bottom = cv::Point(MIN(a.br().x, b.br().x), MIN(a.br().y, b.br().y));
	cv::Point diff = right_bottom - left_top;
	return (MAX(diff.x + 1, 0) * MAX(diff.y + 1, 0));
}

int ComputeIOU(const cv::Rect & rect1,
	const cv::Rect & rect2, float * iou,
	const std::string& type) {

	float inter_area = InterRectArea(rect1, rect2);
	if (type == "UNION") {
		*iou = inter_area / (rect1.area() + rect2.area() - inter_area);
	}
	else {
		*iou = inter_area / MIN(rect1.area(), rect2.area());
	}

	return 0;
}


int NMS(const std::vector<FaceInfo>& faces,
	std::vector<FaceInfo>* result, const float& threshold,
	const std::string& type = "UNION") {
	result->clear();
	if (faces.size() == 0)
		return -1;

	std::vector<size_t> idx(faces.size());

	for (unsigned i = 0; i < idx.size(); i++) {
		idx[i] = i;
	}

	while (idx.size() > 0) {
		int good_idx = idx[0];
		result->push_back(faces[good_idx]);
		std::vector<size_t> tmp = idx;
		idx.clear();
		for (unsigned i = 1; i < tmp.size(); i++) {
			int tmp_i = tmp[i];
			float iou = 0.0f;
			ComputeIOU(faces[good_idx].face_, faces[tmp_i].face_, &iou, type);
			if (iou <= threshold)
				idx.push_back(tmp_i);
		}
	}
}

class BlazeFace::Impl {
public:
    Impl() {
        device_ = 0;
        precision_ = 0;
        power_ = 0;
        memory_ = 0;
        initialized_ = false;
    }
    ~Impl() {

    }

    int LoadModel(const char* root_path);
    int Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces);

    std::shared_ptr<MNN::Interpreter> detector_;
    const int inputSize_ = 128;
    const float X_SCALE = 10.0f;
    const float Y_SCALE = 10.0f;
    const float H_SCALE = 5.0f;
    const float W_SCALE = 5.0f;
    const float ScoreThreshold_ = 0.5f;
    const float NMSThreshold_ = 0.45f;

    int device_;
    int precision_;
    int power_;
    int memory_;


    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;

    bool initialized_;

};

BlazeFace::BlazeFace() {
    impl_ = new BlazeFace::Impl();
}

BlazeFace::~BlazeFace() {
    if (impl_) {
        delete impl_;
    }
}

int BlazeFace::Impl::LoadModel(const char* root_path) {
    std::cout << "start load model." << std::endl;
    std::string model_file = std::string(root_path) + "/blazeface.mnn";
    detector_ = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(model_file.c_str()));
    
    MNN::ScheduleConfig config;
    config.numThread = 1;
    config.type      = static_cast<MNNForwardType>(device_);

    MNN::BackendConfig backendConfig;
    backendConfig.precision = (MNN::BackendConfig::PrecisionMode)precision_;
    backendConfig.power = (MNN::BackendConfig::PowerMode) power_;
    backendConfig.memory = (MNN::BackendConfig::MemoryMode) memory_;
    config.backendConfig = &backendConfig;
    session_ = detector_->createSession(config);

    // nhwc to nchw
    std::vector<int> dims{1, inputSize_, inputSize_, 3};
    input_tensor_ = MNN::Tensor::create<float>(dims, NULL, MNN::Tensor::TENSORFLOW);

    initialized_ = true;

    std::cout << "end load model." << std::endl;


    return 0;
}

int BlazeFace::Impl::Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
    std::cout << "start face detect." << std::endl;
    faces->clear();
    if (!initialized_) {
        std::cout << "model unitialized." << std::endl;
        return 10000;
    }

    // image prepocess
    cv::Mat img_cpy = img_src.clone();
    int width = img_cpy.cols;
    int height = img_cpy.rows;
    float scale_x = static_cast<float>(width) / inputSize_;
    float scale_y = static_cast<float>(height) / inputSize_;

    cv::Mat img_resized;
    cv::resize(img_cpy, img_resized, cv::Size(inputSize_, inputSize_));
    img_resized.convertTo(img_resized, CV_32FC3);
    img_resized = img_resized / 127.5f - 1.0f;

    auto tensor_data = input_tensor_->host<float>();
    auto tensor_size = input_tensor_->size();
    ::memcpy(tensor_data, img_resized.data, tensor_size);

    auto input_tensor = detector_->getSessionInput(session_, nullptr);
    input_tensor->copyFromHostTensor(input_tensor_);
    detector_->runSession(session_);

    // get output
    std::string output_tensor_name0 = "convert_scores";
    std::string output_tensor_name1 = "Squeeze";
    std::string output_tensor_name2 = "anchors";
    MNN::Tensor* tensor_scores  = detector_->getSessionOutput(session_, output_tensor_name0.c_str());
    MNN::Tensor* tensor_boxes   = detector_->getSessionOutput(session_, output_tensor_name1.c_str());
    MNN::Tensor* tensor_anchors = detector_->getSessionOutput(session_, output_tensor_name2.c_str());

    MNN::Tensor tensor_scores_host(tensor_scores, tensor_scores->getDimensionType());
    MNN::Tensor tensor_boxes_host(tensor_boxes, tensor_boxes->getDimensionType());
    MNN::Tensor tensor_anchors_host(tensor_anchors, tensor_anchors->getDimensionType());
    
    tensor_scores->copyToHostTensor(&tensor_scores_host);
    tensor_boxes->copyToHostTensor(&tensor_boxes_host);
    tensor_anchors->copyToHostTensor(&tensor_anchors_host);

    std::cout << "batch:    " << tensor_scores->batch()    << std::endl 
              << "channels: " << tensor_scores->channel()  << std::endl
              << "height:   " << tensor_scores->height()   << std::endl
              << "width:    " << tensor_scores->width()    << std::endl
              << "type:     " << tensor_scores->getDimensionType() << std::endl; 

    // post processing steps
    auto scores_data_ptr  = tensor_scores_host.host<float>();
    auto boxes_data_ptr   = tensor_boxes_host.host<float>();
    auto anchors_data_ptr = tensor_anchors_host.host<float>();

    int OUTPUT_NUM = 960;
    std::vector<FaceInfo> faces_tmp;
    for (int i = 0; i < OUTPUT_NUM; ++i) {
        float center_y = boxes_data_ptr[i * 4 + 0] / Y_SCALE  * anchors_data_ptr[i * 4 + 2] + anchors_data_ptr[i * 4 + 0];
        float center_x = boxes_data_ptr[i * 4 + 1] / X_SCALE  * anchors_data_ptr[i * 4 + 3] + anchors_data_ptr[i * 4 + 1];
        float h    = exp(boxes_data_ptr[i * 4 + 2] / H_SCALE) * anchors_data_ptr[i * 4 + 2];
        float w    = exp(boxes_data_ptr[i * 4 + 3] / W_SCALE) * anchors_data_ptr[i * 4 + 3];

        cv::Rect face = cv::Rect(
            (center_x - 0.5f * w) * width,
            (center_y - 0.5f * h) * height,
            w * width, h * height);
        float score = exp(scores_data_ptr[i * 2 + 1]) /
            (exp(scores_data_ptr[i * 2 + 0]) + exp(scores_data_ptr[i * 2 + 1]));
        
        if (score < ScoreThreshold_) {
            continue;
        }
        FaceInfo face_info;
        face_info.face_ = face;
        face_info.score_ = score;
        faces_tmp.push_back(face_info);
    }
    // std::sort(faces_tmp.begin(), faces_tmp.end(),
	// 	[](const FaceInfo& a, const FaceInfo& b) {
	// 		return a.score_ > b.score_;
	// 	}
    // );
    NMS(faces_tmp, faces, NMSThreshold_);

    std::cout << "end face detect." << std::endl;

    return 0;
}

int BlazeFace::LoadModel(const char* root_path) {
    return impl_->LoadModel(root_path);
}

int BlazeFace::Detect(const cv::Mat& img_src, std::vector<FaceInfo>* faces) {
    return impl_->Detect(img_src, faces);
}