//
//  SizeComputer.cpp
//  MNN
//
//  Created by MNN on 2019/01/10.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SizeComputer.hpp"
#include <stdlib.h>
#include "Macro.h"
#include "TensorUtils.hpp"

namespace MNN {
#ifdef MNN_CODEGEN_REGISTER
void registerShapeOps();
#endif
SizeComputerSuite* SizeComputerSuite::gInstance = nullptr;

SizeComputerSuite::~SizeComputerSuite() {
    for (auto& iter : mRegistry) {
        delete iter.second;
    }
}

SizeComputerSuite* SizeComputerSuite::get() {
    if (nullptr == gInstance) {
        gInstance = new SizeComputerSuite;
#ifdef MNN_CODEGEN_REGISTER
        registerShapeOps();
#endif
    }
    return gInstance;
}

void SizeComputerSuite::insert(SizeComputer* t, OpType type) {
    mRegistry.insert(std::make_pair(type, t));
}

SizeComputer* SizeComputerSuite::search(OpType name) {
    auto iter = mRegistry.find(name);
    if (iter == mRegistry.end()) {
        return nullptr;
    }
    return iter->second;
}
float SizeComputer::onComputeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                   const std::vector<Tensor*>& outputs) const {
    MNN_ASSERT(outputs.size() >= 1);
    return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
}
bool SizeComputer::opNeedContent(OpType type, int index) {
    switch (type) {
        case OpType_Shape:
        case OpType_Rank:
        case OpType_Const:
        case OpType_Size:
        case OpType_PriorBox:
            return false;
        case OpType_Interp:
        case OpType_Crop:
        case OpType_Reshape:
        case OpType_Resize:
            if (1 == index) {
                return false;
            }
            break;
        default:
            break;
    }
    return true;
}
float SizeComputer::computeFlops(const MNN::Op* op, const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    auto computer = computeFactory->search(op->type());
    if (nullptr != computer) {
        return computer->onComputeFlops(op, inputs, outputs);
    }
    return (float)outputs[0]->elementSize() / 1024.0f / 1024.0f;
}

bool SizeComputer::computeOutputSize(const MNN::Op* op, const std::vector<Tensor*>& inputs,
                                     const std::vector<Tensor*>& outputs) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            bool ret = computer->onComputeSize(op, inputs, outputs);
            return ret;
        }
    }

    // Default Set to the same
    if (inputs.size() >= 1 && outputs.size() == 1) {
        if (inputs[0] == outputs[0]) {
            return true;
        }
        const auto& ib = inputs[0]->buffer();
        auto& ob       = outputs[0]->buffer();
        memcpy(ob.dim, ib.dim, sizeof(halide_dimension_t) * ib.dimensions);
        ob.dimensions                                         = ib.dimensions;
        ob.type                                               = ib.type;
        TensorUtils::getDescribe(outputs[0])->dimensionFormat = TensorUtils::getDescribe(inputs[0])->dimensionFormat;
        return true;
    }
    // Not Support
    MNN_PRINT("Can't compute size for %d, name=%s\n", op->type(), op->name()->c_str());

    return false;
}

std::vector<int> SizeComputer::needInputContent(const MNN::Op* op) {
    auto computeFactory = SizeComputerSuite::get();
    // When op is nullptr, it means a copy op
    if (nullptr != op) {
        auto computer = computeFactory->search(op->type());
        if (nullptr != computer) {
            return computer->mNeedContentInputIndex;
        }
    }
    return std::vector<int>{};
}

} // namespace MNN
