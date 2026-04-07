# GE-DPC
Official implementation of "A Granular-Ellipsoid Generation Method Suitable for Density Peak Clustering"

# GE-DPC: A Granular-Ellipsoid Generation Method Suitable for Density Peak Clustering

This repository contains the official Python implementation of the GE-DPC algorithm as described in the paper:

**"A Granular-Ellipsoid Generation Method Suitable for Density Peak Clustering"**

## Abstract
Efficient and robust clustering remains a fundamental challenge in data analysis. While granular-ball (GB)-based methods have enhanced clustering efficiency through granular computing, their spherical structure limits adaptability to non-spherical clusters and boundary representations. Existing granular-ellipsoid (GE) generation methods, however, are primarily designed for classification rather than clustering. To overcome these limitations, we propose a novel GE generation algorithm specifically tailored for density peak clustering (DPC), leading to the development of GE-DPC. Our approach replaces GB with GE to capture complex data distributions better. The GE generation process employs farthest-point pair partitioning refined by Mahalanobis distance optimization, and introduces a mechanism for identifying and re-segmenting outlier GEs based on a density-driven quality criterion. This effectively refines boundary handling and prevents over-segmentation. Moreover, we redefine local density and minimum distance at the GE granularity, enabling clustering at the GE level. Extensive experiments on synthetic and real-world datasets demonstrate that while spherical methods may excel in specific isotropic scenarios, GE-DPC exhibits superior robustness and adaptability across diverse data distributions. Quantitatively, GE-DPC achieves the highest average accuracy (ACC), normalized mutual information (NMI), and adjusted rand index (ARI) compared to state-of-the-art GB-based methods, while recording the lowest performance variance, underscoring its exceptional stability and distinct advantages in handling anisotropic clusters.
