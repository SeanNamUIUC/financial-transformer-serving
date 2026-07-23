//중복 정의 에러 차단목적
#pragma once

#include<vector>
//c 시스템 메모리 크기나 주소관련 정의들을 모아놓음. size_t -> 시스템이 가질수 있는 최대 메모리 크기 를 unsigned int 로 표현하도록 하드웨어 맞춤 설계 타입
#include<cstddef>
#include<iostream>
#include<cassert>

class Tensor {
    public:
        Tensor (const std::vector<size_t>& shape);//constructor
        ~Tensor(); //destructor


        //copy constructor prevention -> to prvent doulbe free error 
        Tensor(const Tensor&) = delete;
        Tensor& operator=(const Tensor&) = delete;


        // std::move constructor and move operator is available
        Tensor(Tensor&& other) noexcept;
        Tensor& operator=(Tensor&& other) noexcept;


        //Hardware accssing interface
        //front const: no change for parameter we've taken vs back const: no revision for inner member variable
        float* data() { return data_ptr_; }
        const float* data() const { return data_ptr_; }

        const std::vector<size_t>& shape() const { return shape_; }
        const std::vector<size_t>& strides() const { return strides_; }
        size_t size() const { return total_size_; }

        void print_metadata() const;
    private:
        float* data_ptr_; //starting pointer of vram
        std::vector<size_t> shape_; //store each dimension ex) in our case: [32, 20, 8]
        std::vector<size_t> strides_; //A coefficient that contains the distance to skip in order to move to the next element in memory
        size_t total_size_;  // total number of elements

        void compute_strides();//computing strides based on shape
};

