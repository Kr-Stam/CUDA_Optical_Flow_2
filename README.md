# Repository Moved

**Notice:**
This GitHub repository is no longer actively maintained. Development
for this project has officially moved to Codeberg.

Please find the active repository, latest updates, and
current CUDA kernels here: [https://codeberg.org/Kr-Stam/StreamFlow]

**Note:** The new repository is a full refactor of the
original code and follows a different design philosophy.

--------------------------------------------------------------------------------
## About This Project

This repository originally hosted the source code corresponding
to the research paper "Speeding up Dense Optical Flow
Estimation with CUDA" by Kristijan Stameski and Marjan
Gusev.
**DOI:** [10.1109/TELFOR63250.2024.10819107](https://doi.org/10.1109/TELFOR63250.2024.10819107)

This project implements and optimizes the pyramidal Lucas-Kanade optical flow
estimation algorithm by dividing the algorithm into discrete steps
to achieve massive performance boosts and real-time execution.

Key features of the codebase include:

 - Gaussian Pyramid Integration: Sub-sampling images to solve the
   Lucas-Kanade method's flaw of disregarding significant movements.

 - CUDA Optimizations: GPU-accelerated implementations of bilinear
   filters, convolutions (for partial derivatives), and sum reduction
   over a window, making the total execution time up to five times
   faster than CPU execution.
