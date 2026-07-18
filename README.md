## Software Directory Structure & Roadmap

현재 구축 완료된(Production-ready) **PyTorch 기반 딥러닝 파이프라인(1트랙)**과 향후 극강의 성능 가속화를 위해 구현 예정인 **모던 C++/CUDA 기반 가속기 엔진(2트랙)의 최종 설계도**입니다.
This provides the definitive blueprint combining the production-ready **PyTorch Deep Learning Pipeline (Track 1)** and the upcoming **Modern C++/CUDA Acceleration Engine (Track 2)** designed for ultra-low latency inference.

```text
financial-transformer-serving/
├── .venv/                         # Codespace (Python Virtual Environment)
├── accelerator/                   # [Track 2: High-Performance C++ Hardware Acceleration Layer]
│   ├── CMakeLists.txt             # C++ build script (Supports Modern C++17/20 & CUDA compilation)
│   ├── model_export.py            # Weights extractor & INT8 Quantization utility from PyTorch to Binary
│   ├── include/                   # Header files for memory map & custom kernels
│   │   ├── tensor.hpp             # High-performance 1D Row-major physical contiguous memory tensor
│   │   ├── gemm_accelerator.hpp   # [Targeting Compute-bound] CUDA kernel headers for massively parallel General Matrix Multiplication
│   │   ├── fusion_engine.hpp      # [Targeting Memory-bound] Operator Fusion engine merging element-wise operations (sqrt, add, mul)
│   │   ├── dma_channel.hpp        # [Targeting I/O Overhead] Virtual channel interface for Host-to-Device Zero-Copy DMA memory mapping
│   │   └── quantization.hpp       # Scaling and clipping utilities optimized for accelerated INT8 inference
│   └── src/                       # Source implementations for the core acceleration engines
│       ├── tensor.cpp             # PyTorch tensor binding to raw C++ pointers and contiguous memory management
│       ├── gemm_accelerator.cu    # Fast GEMM implementation via CUDA thread block scheduling and Tensor Core utilization 
│       ├── fusion_engine.cu       # Operator Fusion implementation 
│       ├── dma_channel.cpp        # Zero-Copy memory mapping logic to completely bypass explicit cudaMemcpy data transfer latencies
│       └── main.cpp               # Benchmarking and latency profiling demonstration script with financial data injection
├── config/ 
│   └── config.yaml                # Data pipeline configurations and hyperparameter paths
├── data/
│   └── financial_market.db        # Collected SQLite financial database (Isolated via .gitignore for scale)
├── docs/                          # [Deep-Dive Technical Documentations]
│   ├── 01_concept_and_architecture.md    # Financial-specific 24-Layer Transformer architecture design
│   ├── 02_training_and_optimization.md   # PyTorch VRAM optimization, Mixed Precision (AMP), & memory tuning
│   └── 03_troubleshooting.md             # Hybrid workflow (Codespace ⇄ Colab) & hardware mismatch fixes
├── src/                           # [Track 1: PyTorch Deep Learning Production Pipeline]
│   ├── data_pipeline/             # Raw data collection pipeline
│   │   └── collector.py           # Financial data scraper & SQLite pipeline injector
│   └── model/                     # Customized Transformer Model
│       ├── attention.py           # Custom Multi-Head Attention featuring Causal Masking mechanisms
│       ├── dataset.py             # SQLite data-fetching pipeline and sliding window generator
│       ├── model_main.py          # Main orchestrator for system integration and training trigger
│       ├── trainer.py             # Clean training engine executing 1-epoch loop operations
│       └── transformer_block.py   # Layer definitions for the 24-stacked Transformer Encoder blocks
└── tests/                         # Unit Testing files for system stability
    └── test_attention.py          # Validating causal masking dimensionality correctness and attention calculation
