# Changelog
All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).


## [0.1.0] - 2025-10-05 - Beta release

This marks the transition of Quantreo into **Beta stage (v0.1.0)**. 
No new features were added, but this version represents a major step toward a stable and ready-to-use 1.0.0 release.  
The focus of this release is **structure, reliability, and developer experience**.

- **Added:** Integration of essential `raise` error checks within multiple core functions to provide clearer and safer feedback to users during incorrect usage or parameter mismatch.
- **Changed:** Major internal refactor to improve **code structure** and maintainability across all modules.  
- **Changed:** Enhanced and reorganized **documentation** for better clarity and navigation.  
- **Changed:** Improved **unit test coverage** across the library to ensure consistency and long-term stability.  
- **Changed:** Updated `log_returns` function, added explicit error handling for invalid inputs, improved numerical stability, and refined code readability.


## [0.0.30] - 2025-09-24
- **Added**: 6 new feature engineering functions: causal fourier reconstruction, causal wavelet reconstruction, logit transform, negative log transform, bimodality coefficient, market Mode Average (MMA).
- **Changed**: Reorganized codebase by splitting each major package (`features`, `targets`, `data`) into dedicated thematic submodules/files.


## [0.0.24 – 0.0.29] - 2025-05 → 2025-07
- **Added**: Data aggregation package (in order to create bars from ticks)
- **Added**: Feature engineering functions


## [0.0.13 – 0.0.23] - 2025-03 → 2025-05
- **Added**: Feature engineering functions
- **Added**: Target engineering package


## [0.0.1 – 0.0.12] - 2025-03
- **Added**: Features engineering package
- **Added**: Initial development & repository setup.

