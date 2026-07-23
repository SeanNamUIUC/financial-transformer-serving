#include "include/tensor.hpp"
#include <iostream>
#include <cassert>


//Testing:clang++ -std=c++17 -Iaccelerator -o test_tensor accelerator/tests/test_tensor.cpp accelerator/src/tensor.cpp
//./test_tensor
int main() {
    std::cout << "[Test 1] Creating Tensor" << std::endl;
    
    //[2,3]
    Tensor t({2, 3});
    
    
    const auto& shape = t.shape();
    const auto& strides = t.strides();
    
    assert(shape[0] == 2 && shape[1] == 3);
    assert(strides[0] == 3 && strides[1] == 1);
    
    std::cout << "shape and strided test completed" << std::endl;
    //printing out metadata
    t.print_metadata();


    std::cout << "=========================================" << std::endl;
    std::cout << "  finished testing  " << std::endl;
    std::cout << "=========================================" << std::endl;

    return 0;
}