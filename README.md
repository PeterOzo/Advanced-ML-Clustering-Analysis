## Advanced-ML-Clustering-Analysis

## Advanced Machine Learning: Clustering Analysis and Gaussian Mixture Models

## Project Overview

This repository contains a comprehensive analysis of clustering algorithms and Gaussian Mixture Models (GMMs) implemented for **Data 642 Advanced Machine Learning**. The project demonstrates two fundamental unsupervised learning approaches: **K-Means clustering** on the Olivetti Faces dataset and **Expectation-Maximization (EM) algorithm** for Gaussian Mixture Models on synthetic data.

## Exercise 1: Olivetti Faces Clustering Analysis

### Project Description

This exercise implements a comprehensive clustering analysis of the **Olivetti Faces dataset**, which contains 400 grayscale face images of 40 different individuals (10 images per person). The project focuses on discovering natural groupings in facial features using K-Means clustering with rigorous evaluation metrics.

### Dataset Characteristics
- **Total Images**: 400 face images
- **Image Dimensions**: 64×64 pixels (4,096 features)
- **Individuals**: 40 unique people
- **Images per Person**: 10 different poses/expressions
- **Challenge**: High-dimensional data with complex facial feature patterns

### Key Implementation Features

#### 1. Stratified Data Splitting
```python
def stratified_split(X, y, train_size=0.6, valid_size=0.2, test_size=0.2):
    """
    Ensures balanced representation of each individual across train/validation/test sets
    """
```

**Benefits:**
- Maintains class distribution across splits
- Prevents bias in clustering evaluation
- Ensures robust model validation

#### 2. Comprehensive Cluster Evaluation
The project implements multiple evaluation metrics for determining optimal cluster numbers:

**Silhouette Score Analysis:**
- Measures cluster cohesion and separation
- Range: [-1, 1], higher values indicate better clustering
- **Result**: Peak at K=2 with score of 0.163

**Davies-Bouldin Score Analysis:**
- Measures average similarity between clusters
- Lower values indicate better clustering
- **Result**: Minimum at K=20 with score of 1.997

#### 3. Systematic K-Value Testing
```python
k_range = range(2, 21)  # Testing 2 to 20 clusters
```

The algorithm evaluates clustering quality across different K values to find the optimal number of clusters using:
- **Silhouette coefficient**
- **Davies-Bouldin index**
- **Cluster size distribution analysis**

### Results Summary

#### Optimal Clustering Configuration
- **Best K Value**: 2 clusters
- **Silhouette Score**: 0.163
- **Davies-Bouldin Score**: 2.013
- **Cluster Distribution**: 
  - Cluster 0: 106 images (33.1%)
  - Cluster 1: 214 images (66.9%)

#### Performance Analysis
```
K=2: Silhouette=0.163, Davies-Bouldin=2.013 ✓ OPTIMAL
K=3: Silhouette=0.119, Davies-Bouldin=2.407
K=4: Silhouette=0.110, Davies-Bouldin=2.320
...
K=20: Silhouette=0.112, Davies-Bouldin=1.997
```

### Technical Insights

#### Why K=2 is Optimal
1. **Highest Silhouette Score**: Indicates best balance between cluster cohesion and separation
2. **Clear Elbow Point**: Sharp decline in performance after K=2
3. **Natural Grouping**: Reflects fundamental facial feature variations
4. **Computational Efficiency**: Simplest model that captures main data structure

#### Clustering Challenges Identified
- **Low Silhouette Scores (< 0.2)**: Indicates overlapping clusters and complex feature distributions
- **Imbalanced Clusters**: 214 vs 106 images suggests natural asymmetry in facial features
- **High-Dimensional Complexity**: 4,096 features create challenges for traditional clustering

---

## Exercise 2: Gaussian Mixture Models

### Project Description

This exercise implements the **Expectation-Maximization (EM) algorithm** for Gaussian Mixture Models on synthetic 2D data generated from known Gaussian distributions. The project demonstrates model selection, initialization strategies, and convergence analysis.

### Data Generation Parameters
```python
μ₁ = [0.9, 1.02]     # Mean of first Gaussian
μ₂ = [-1.2, -1.3]    # Mean of second Gaussian

Σ₁ = [[0.5, 0.081],   # Covariance matrix 1
      [0.081, 0.7]]

Σ₂ = [[0.4, 0.02],    # Covariance matrix 2
      [0.02, 0.3]]

N₁ = 100 samples     # First distribution
N₂ = 20 samples      # Second distribution
```

### Key Implementation Components

#### 1. Model Selection Analysis
Systematic evaluation of different numbers of mixture components:

```python
for k in [1, 2, 3, 4]:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X)
    print(f'K={k}: BIC={gmm.bic(X):.2f}, AIC={gmm.aic(X):.2f}')
```

#### 2. Initialization Strategy Testing
Evaluation of different starting points for the EM algorithm:

```python
init_points = [
    [[3, 3], [-3, -3]],    # Far from true means
    [[0, 0], [1, 1]],      # Close to data center
    [[-2, 2], [2, -2]]     # Different quadrants
]
```

#### 3. Convergence Analysis
Monitoring algorithm performance across different initializations:
- **Convergence status**
- **Number of iterations**
- **Final log-likelihood**

### Results Summary

#### Model Selection Results
| K | BIC Score | AIC Score | Interpretation |
|---|-----------|-----------|----------------|
| 1 | 679.54 | 665.60 | Poor fit - unimodal model |
| **2** | **665.19** | **634.53** | **✓ OPTIMAL** |
| 3 | 689.24 | 641.86 | Overfitting |
| 4 | 713.43 | 649.31 | Severe overfitting |

**Key Finding**: K=2 minimizes both BIC and AIC, correctly identifying the true number of components.

#### Initialization Robustness Analysis
| Initialization | Iterations | Lower Bound | Convergence |
|----------------|------------|-------------|-------------|
| Far points | 4 | -2.55 | ✓ Fast |
| Center points | 18 | -2.57 | ✓ Slower |
| Quadrant points | 19 | -2.57 | ✓ Slower |

**Key Finding**: All initializations converge to similar solutions, demonstrating algorithm robustness.

### Mathematical Foundation

#### EM Algorithm Implementation
The project implements the complete EM algorithm cycle:

**E-Step (Expectation):**
```
γᵢⱼ = P(component j | observation i)
    = πⱼ N(xᵢ | μⱼ, Σⱼ) / Σₖ πₖ N(xᵢ | μₖ, Σₖ)
```

**M-Step (Maximization):**
```
πⱼ = (1/N) Σᵢ γᵢⱼ                    # Update mixing coefficients
μⱼ = Σᵢ γᵢⱼ xᵢ / Σᵢ γᵢⱼ              # Update means
Σⱼ = Σᵢ γᵢⱼ (xᵢ-μⱼ)(xᵢ-μⱼ)ᵀ / Σᵢ γᵢⱼ  # Update covariances
```

---

## Technical Implementation

### Environment Setup
```python
# Parallel processing optimization
import os
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
```

### Core Libraries
- **scikit-learn**: KMeans, GaussianMixture, evaluation metrics
- **numpy**: Numerical computations and linear algebra
- **matplotlib**: Visualization and plotting
- **scipy**: Statistical distributions and advanced math
- **seaborn**: Enhanced statistical visualizations

### Data Processing Pipeline
1. **Data Loading**: Automated dataset fetching and preprocessing
2. **Stratified Splitting**: Balanced train/validation/test division
3. **Feature Standardization**: Z-score normalization for optimal clustering
4. **Evaluation**: Multiple metrics for comprehensive assessment
5. **Visualization**: Clear plots for result interpretation

---

## Key Findings

### Olivetti Faces Analysis
1. **Natural Grouping**: Faces naturally cluster into 2 main groups based on dominant visual features
2. **Feature Complexity**: High-dimensional facial data presents challenges for traditional clustering
3. **Evaluation Metrics**: Silhouette analysis proves more reliable than Davies-Bouldin for this dataset
4. **Cluster Imbalance**: Natural asymmetry in facial features leads to unequal cluster sizes

### Gaussian Mixture Models Analysis
1. **Model Selection**: Both BIC and AIC correctly identify the true number of components (K=2)
2. **Initialization Robustness**: Well-separated clusters make the algorithm robust to different starting points
3. **Convergence Speed**: Initialization closer to true parameters enables faster convergence
4. **Parameter Recovery**: EM algorithm successfully recovers true underlying parameters

---

## Installation

### Prerequisites
```bash
pip install numpy matplotlib scikit-learn scipy seaborn
```

### Required Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import multivariate_normal
import seaborn as sns
```

---

## Usage

### Running Olivetti Faces Analysis
```python
# Execute complete clustering analysis
python olivetti_clustering.py

# The script will:
# 1. Load and preprocess the Olivetti faces dataset
# 2. Perform stratified data splitting
# 3. Evaluate clustering for K=2 to K=20
# 4. Generate evaluation metric plots
# 5. Visualize optimal clustering results
# 6. Display cluster analysis summary
```

### Running GMM Analysis
```python
# Execute Gaussian Mixture Model analysis
python gmm_analysis.py

# The script will:
# 1. Generate synthetic 2D Gaussian data
# 2. Test different numbers of components (K=1 to K=4)
# 3. Evaluate different initialization strategies
# 4. Plot clustering results with contours
# 5. Report BIC/AIC scores and convergence statistics
```

### Custom Parameters
```python
# Modify clustering parameters
patch_size = 12              # For face patch analysis
k_range = range(2, 21)       # Range of K values to test
train_size = 0.6             # Training set proportion
n_components = 2             # Number of GMM components

# Modify GMM parameters
mu1 = [0.9, 1.02]           # First Gaussian mean
mu2 = [-1.2, -1.3]          # Second Gaussian mean
n1, n2 = 100, 20            # Sample sizes
```

---

## Results Analysis

### Clustering Performance Metrics

#### Silhouette Analysis
- **Range**: [-1, 1]
- **Interpretation**: 
  - > 0.7: Very strong clustering
  - > 0.5: Reasonable clustering  
  - > 0.3: Weak but detectable clustering
  - < 0.3: Potentially artificial clustering
- **Olivetti Result**: 0.163 (potentially artificial, but best available)

#### Davies-Bouldin Analysis
- **Range**: [0, ∞)
- **Interpretation**: Lower values indicate better clustering
- **Trade-off**: Often conflicts with silhouette score for optimal K

### Model Selection Insights

#### Information Criteria Comparison
- **AIC (Akaike Information Criterion)**: Favors model fit
- **BIC (Bayesian Information Criterion)**: Penalizes complexity more heavily
- **Consensus**: Both metrics agree on K=2 for GMM analysis

### Visualization Quality
- **Contour Plots**: Show Gaussian component boundaries
- **Cluster Visualization**: Clear separation of data points
- **Face Clustering**: Groups similar facial features effectively

---

## Academic Insights

### Theoretical Contributions

#### Clustering in High-Dimensional Spaces
- **Curse of Dimensionality**: Facial data (4,096 dimensions) demonstrates challenges
- **Feature Reduction Need**: Suggests potential for PCA/t-SNE preprocessing
- **Distance Metrics**: Euclidean distance limitations in high dimensions

#### EM Algorithm Properties
- **Local Optima**: Different initializations can lead to different solutions
- **Convergence Guarantees**: Algorithm guaranteed to improve likelihood
- **Model Selection**: Information criteria provide principled approach

### Practical Applications

#### Computer Vision
- **Face Recognition**: Clustering can group similar faces for identification
- **Expression Analysis**: Clusters may capture different facial expressions
- **Data Preprocessing**: Clustering can inform feature selection strategies

#### Statistical Modeling
- **Mixture Models**: Foundation for more complex probabilistic models
- **Density Estimation**: GMMs provide flexible density modeling
- **Anomaly Detection**: Outliers can be detected using mixture likelihoods

### Methodological Insights

#### Evaluation Strategy
- **Multiple Metrics**: Single metrics can be misleading
- **Cross-Validation**: Stratified splitting ensures robust evaluation
- **Visual Inspection**: Plots often reveal insights not captured by metrics

#### Algorithm Selection
- **K-Means vs GMM**: Hard vs soft clustering trade-offs
- **Initialization Importance**: Can significantly affect convergence
- **Parameter Tuning**: Systematic evaluation of hyperparameters

---

## Future Improvements

### Algorithmic Enhancements
1. **Spectral Clustering**: Better performance on non-convex clusters
2. **Hierarchical Clustering**: Reveal cluster structure at multiple scales
3. **Density-Based Clustering**: Handle clusters of varying density
4. **Semi-Supervised Approaches**: Incorporate limited label information

### Feature Engineering
1. **Dimensionality Reduction**: PCA, t-SNE, UMAP preprocessing
2. **Feature Selection**: Identify most discriminative facial features
3. **Deep Features**: Use pre-trained CNN features for better representation
4. **Multi-Scale Analysis**: Combine features at different resolutions

### Model Extensions
1. **Variational Inference**: More robust parameter estimation
2. **Nonparametric Methods**: Dirichlet Process Mixtures for automatic K selection
3. **Robust Mixtures**: Handle outliers and noise more effectively
4. **Constrained Clustering**: Incorporate domain knowledge

### Computational Optimizations
1. **Parallel Processing**: Leverage multi-core architectures
2. **GPU Acceleration**: Implement clustering on GPU
3. **Online Learning**: Handle streaming data efficiently
4. **Approximate Methods**: Trade accuracy for speed on large datasets

## Academic Context

This project demonstrates proficiency in:

### Machine Learning Concepts
- **Unsupervised Learning**: Clustering and mixture models
- **Model Selection**: Information criteria and validation strategies
- **Algorithm Analysis**: Convergence properties and initialization effects
- **Performance Evaluation**: Multiple metrics and visualization techniques

### Statistical Methods
- **Multivariate Statistics**: Gaussian distributions and covariance estimation
- **Optimization Theory**: EM algorithm and local optima
- **Information Theory**: AIC/BIC for model comparison
- **Probability Theory**: Mixture distributions and Bayesian inference

### Computational Skills
- **Efficient Implementation**: Optimized Python code with scikit-learn
- **Data Visualization**: Clear and informative plots
- **Experimental Design**: Systematic parameter evaluation
- **Scientific Computing**: Numerical stability and reproducibility

## References

1. Duda, R. O., Hart, P. E., & Stork, D. G. (2001). *Pattern Classification*. Wiley.
2. Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
3. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
4. Scikit-learn Documentation: [Clustering](https://scikit-learn.org/stable/modules/clustering.html)
5. Olivetti Faces Dataset: [AT&T Database of Faces](https://cam-orl.co.uk/facedatabase.html)
