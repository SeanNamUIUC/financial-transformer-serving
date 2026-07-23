#include "include/tensor.hpp"


// Constructor
Tensor::Tensor(const std::vector<size_t>& shape) {
    data_ptr_ = nullptr;
    shape_ = shape;
    total_size_ = 1; // preventing total_size 0 for future total_size calculation

    // Defensive programming: empty shape check
    if (shape_.empty()) {
        throw std::invalid_argument("[Error] Tensor shape is empty! Dimension size must be >= 1.");
    }
    
    // Calculate total_size_
    for (size_t v : shape_) {
        if (v == 0) {
            throw std::invalid_argument("[Error] Tensor dimension size cannot be 0.");
        }
        total_size_ *= v;
    }
    
    // heap memory allocation
    data_ptr_ = new float[total_size_];

    // Computing strides
    compute_strides();
}

// Destructor
Tensor::~Tensor() {
    if (data_ptr_ != nullptr) {
        delete[] data_ptr_;
        data_ptr_ = nullptr;
    }
}

// Compute Strides
void Tensor::compute_strides() {
    size_t shape_size = shape_.size();
    strides_.resize(shape_size);

    // last dimension is always 1
    strides_[shape_size - 1] = 1;

    // strides computation
    for (size_t i = shape_size - 1; i > 0; --i) {
        strides_[i - 1] = shape_[i] * strides_[i];
    }
}


void Tensor::print_metadata() const {
    std::cout << "=== Tensor Metadata ===" << std::endl;
    
    // print shape
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape_.size(); ++i) {
        std::cout << shape_[i] << (i + 1 < shape_.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;

    // 2. print strides
    std::cout << "Strides: [";
    for (size_t i = 0; i < strides_.size(); ++i) {
        std::cout << strides_[i] << (i + 1 < strides_.size() ? ", " : "");
    }
    std::cout << "]" << std::endl;

    // 3. print additional information
    std::cout << "Total Elements: " << total_size_ << std::endl;
    std::cout << "Data Pointer: " << static_cast<void*>(data_ptr_) << std::endl;

    
    std::cout << "=======================" << std::endl;
}