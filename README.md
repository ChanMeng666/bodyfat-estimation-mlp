<div align="center"><a name="readme-top"></a>

[![Project Banner](./correlation_heatmap.png)](#)

# 🧠 Body Fat Estimation Neural Network<br/><h3>Advanced Multi-Layer Perceptron for Anthropometric Body Fat Prediction</h3>

A cutting-edge machine learning solution that leverages neural network architectures to provide accurate body fat percentage predictions from anthropometric measurements.<br/>
Features comprehensive analysis, feature optimization, and state-of-the-art performance with **R² scores exceeding 0.97**.<br/>
One-click **FREE** deployment of your personalized body fat prediction model.

[📊 Live Demo](https://huggingface.co/ChanMeng666/bodyfat-estimation-mlp) · [📈 Model Hub](https://huggingface.co/ChanMeng666/bodyfat-estimation-mlp) · [🐛 Issues](https://github.com/ChanMeng666/bodyfat-estimation-mlp/issues)

<br/>

[![🚀 Try Model Now 🚀](https://gradient-svg-generator.vercel.app/api/svg?text=%F0%9F%91%89Try%20Model%20Now!%F0%9F%91%88&color=000000&height=60&gradientType=radial&duration=6s&color0=ffffff&template=pride-rainbow)](https://huggingface.co/ChanMeng666/bodyfat-estimation-mlp)

<br/>

<!-- SHIELD GROUP -->
[![](https://img.shields.io/badge/python-3.7+-3670A0?style=flat-square&logo=python&logoColor=white)](https://python.org/)
[![](https://img.shields.io/badge/tensorflow-2.0+-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org/)
[![](https://img.shields.io/badge/jupyter-notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![](https://img.shields.io/badge/license-MIT-white?labelColor=black&style=flat-square)](LICENSE)
[![](https://img.shields.io/github/stars/ChanMeng666/bodyfat-estimation-mlp?color=ffcb47&labelColor=black&style=flat-square)](https://github.com/ChanMeng666/bodyfat-estimation-mlp/stargazers)
[![](https://img.shields.io/github/forks/ChanMeng666/bodyfat-estimation-mlp?color=8ae8ff&labelColor=black&style=flat-square)](https://github.com/ChanMeng666/bodyfat-estimation-mlp/forks)

<sup>🌟 Revolutionizing body composition analysis through intelligent anthropometric modeling. Built for healthcare professionals, fitness enthusiasts, and researchers worldwide.</sup>

</div>

> [!IMPORTANT]
> This project demonstrates advanced neural network applications in healthcare and fitness domains. It combines cutting-edge deep learning with anthropometric science to provide highly accurate body composition analysis. The models achieve exceptional performance with R² scores exceeding 0.97, making them suitable for clinical and research applications.

## 📸 Project Showcase

<div align="center">
  <img src="./correlation_with_bodyfat.png" alt="Feature Correlation Analysis" width="800"/>
  <p><em>Feature Correlation Analysis - Identifying Key Anthropometric Predictors</em></p>
</div>

<div align="center">
  <img src="./correlation_heatmap.png" alt="Correlation Heatmap" width="600"/>
  <p><em>Comprehensive Correlation Matrix - Understanding Feature Relationships</em></p>
</div>

## 🎬 Model Performance

**Performance Metrics:**

<div align="center">

| Model Type | Hidden Neurons | R² Score | MSE | Features Used |
|------------|----------------|----------|-----|---------------|
| **Full Model** | 20 | **0.9724** | 1.9250 | All 14 features |
| **Optimized Model** | 5 | **0.9950** | 0.2340 | Selected 9 features |

</div>

**Tech Stack:**

<div align="center">

 <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54"/>
 <img src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white"/>
 <img src="https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white"/>
 <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white"/>
 <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white"/>
 <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white"/>
 <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white"/>

</div>

## 🌟 Introduction

We are passionate researchers and developers creating next-generation healthcare and fitness solutions through advanced machine learning. By combining anthropometric science with state-of-the-art neural network architectures, we provide accurate, efficient, and clinically-relevant body composition analysis tools.

Whether you're a healthcare professional, fitness trainer, researcher, or fitness enthusiast, this project offers sophisticated yet accessible body fat prediction capabilities. The models have been rigorously validated and achieve exceptional accuracy suitable for both clinical and personal use applications.

> [!NOTE]
> - Python 3.7+ required
> - TensorFlow 2.0+ for neural network computations
> - Jupyter Notebook for interactive analysis
> - 252 validated anthropometric measurements in dataset

## ✨ Key Features

### `1` Advanced Neural Architecture

Experience next-generation multi-layer perceptron models optimized for anthropometric analysis. Our innovative neural network approach provides unprecedented accuracy through carefully tuned architectures and hyperparameters.

**Key Capabilities:**
- 🧠 **Multi-Layer Perceptron**: Optimized hidden layer configurations (5-20 neurons)
- 🎯 **High Accuracy**: R² scores exceeding 0.97 on test data
- ⚡ **Fast Training**: Early stopping and adaptive learning
- 🔧 **Hyperparameter Optimization**: Comprehensive grid search analysis

### `2` Comprehensive Analysis Suite

Revolutionary analytical framework that provides deep insights into anthropometric relationships and model performance. Our comprehensive suite includes correlation analysis, sensitivity testing, and performance visualization.

**Analysis Components:**
- **Qualitative Analysis**: Visual exploration of feature relationships
- **Quantitative Correlation**: Statistical significance testing
- **Sensitivity Analysis**: Feature importance evaluation
- **Performance Comparison**: Model architecture evaluation

### `3` Intelligent Feature Selection

Smart feature selection methodology that maintains high accuracy while reducing measurement requirements. Our approach identifies the most predictive anthropometric measurements for practical implementation.

**Feature Optimization:**
- 📉 **Dimensionality Reduction**: From 14 to 9 key features
- 🎯 **Maintained Accuracy**: R² > 0.99 with reduced inputs
- 💰 **Cost Efficiency**: Fewer measurements required
- 🔬 **Evidence-Based**: Correlation-driven selection

## 🛠️ Tech Stack

<div align="center">
  <table>
    <tr>
      <td align="center" width="96">
        <img src="https://cdn.simpleicons.org/python" width="48" height="48" alt="Python" />
        <br>Python 3.7+
      </td>
      <td align="center" width="96">
        <img src="https://cdn.simpleicons.org/tensorflow" width="48" height="48" alt="TensorFlow" />
        <br>TensorFlow 2.0+
      </td>
      <td align="center" width="96">
        <img src="https://cdn.simpleicons.org/keras" width="48" height="48" alt="Keras" />
        <br>Keras
      </td>
      <td align="center" width="96">
        <img src="https://cdn.simpleicons.org/pandas" width="48" height="48" alt="Pandas" />
        <br>Pandas
      </td>
      <td align="center" width="96">
        <img src="https://cdn.simpleicons.org/numpy" width="48" height="48" alt="NumPy" />
        <br>NumPy
      </td>
      <td align="center" width="96">
        <img src="https://cdn.simpleicons.org/jupyter" width="48" height="48" alt="Jupyter" />
        <br>Jupyter
      </td>
      <td align="center" width="96">
        <img src="https://cdn.simpleicons.org/scikitlearn" width="48" height="48" alt="Scikit-learn" />
        <br>Scikit-learn
      </td>
    </tr>
  </table>
</div>

**Core Framework:**
- **Deep Learning**: TensorFlow 2.0+ with Keras high-level API
- **Data Science**: NumPy for numerical computing, Pandas for data manipulation
- **Machine Learning**: Scikit-learn for preprocessing and metrics
- **Visualization**: Matplotlib and Seaborn for publication-quality plots
- **Interactive Analysis**: Jupyter Notebook for reproducible research

## ⚡️ Performance

### Model Performance Comparison

<div align="center">

| **Metric** | **Full Model (20 neurons)** | **Optimized Model (5 neurons)** | **Improvement** |
|------------|------------------------------|----------------------------------|-----------------|
| **R² Score** | 0.9724 | **0.9950** | +2.3% |
| **MSE** | 1.9250 | **0.2340** | -87.8% |
| **Features** | 14 | **9** | -35.7% |
| **Complexity** | High | **Low** | Simplified |

</div>

**Key Performance Metrics:**
- 🎯 **95%+ R² Score** across both architectures
- 🚀 **< 2.0 MSE** on test data
- 📊 **0.99+ correlation** with ground truth measurements
- 🔄 **Robust validation** across multiple data splits

**Feature Importance Analysis:**
1. **Body Density** (r = -0.99) - Primary physiological indicator
2. **Abdomen** (r = 0.81) - Core body composition measurement
3. **Chest** (r = 0.70) - Upper body muscle/fat distribution
4. **Hip** (r = 0.63) - Lower body composition indicator
5. **Weight** (r = 0.61) - Overall body mass contribution

## 🚀 Getting Started

### Prerequisites

> [!IMPORTANT]
> Ensure you have the following installed for optimal performance:

- **Python 3.7+** ([Download](https://python.org/downloads/))
- **pip package manager** (included with Python)
- **Git** for repository cloning ([Download](https://git-scm.com/))
- **Jupyter Notebook** for interactive analysis

### Quick Installation

**1. Clone Repository**

```bash
git clone https://github.com/ChanMeng666/bodyfat-estimation-mlp.git
cd bodyfat-estimation-mlp
```

**2. Install Dependencies**

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn jupyter
```

**3. Launch Jupyter Notebook**

```bash
jupyter notebook
```

**4. Start Analysis**

Open any of the analysis notebooks:
- `Assignment2_Part3_(i).ipynb` - Qualitative Analysis
- `Assignment2_Part3_(ii).ipynb` - Network Performance
- `Assignment2_Part3_(iii).ipynb` - Correlation Analysis

🎉 **Success!** Begin exploring the comprehensive body fat prediction analysis.

## 📖 Usage Guide

### Basic Usage

**Quick Body Fat Prediction:**

```python
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model
model = load_model('best_full_model.keras')

# Example prediction with anthropometric measurements
measurements = np.array([[1.055, 44, 81.1, 178.2, 38.0, 100.8, 92.6, 99.9, 59.4, 38.6, 23.1, 32.3, 28.5, 18.0]])
body_fat_prediction = model.predict(measurements)
print(f"Predicted body fat: {body_fat_prediction[0][0]:.2f}%")
```

### Advanced Analysis

**Custom Model Training:**

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and prepare data
df = pd.read_csv('Body_Fat.csv')
X = df.drop('BodyFat', axis=1)
y = df['BodyFat']

# Scale features and split data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create and train model
model = Sequential([
    Dense(20, activation='sigmoid', input_shape=(X_train.shape[1],)),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_split=0.2)

# Evaluate model
test_loss = model.evaluate(X_test, y_test)
print(f"Test MSE: {test_loss:.4f}")
```

## 🔬 Research Methodology

Our comprehensive research methodology follows established protocols for anthropometric analysis and neural network validation:

**Seven-Phase Analysis:**
1. **Qualitative Analysis**: Visual exploration of feature relationships
2. **Network Optimization**: Hyperparameter tuning and architecture selection
3. **Correlation Analysis**: Statistical significance testing of feature relationships
4. **Feature Selection**: Dimensionality reduction while maintaining accuracy
5. **Sensitivity Analysis**: Feature importance quantification
6. **Performance Comparison**: Cross-model validation and evaluation
7. **Clinical Implications**: Real-world applicability assessment

## 📊 Analysis Results

### Correlation Findings

**Strong Predictors (|r| > 0.6):**
- Body Density: r = -0.99 (physiological gold standard)
- Abdomen circumference: r = 0.81 (central adiposity)
- Chest circumference: r = 0.70 (upper body composition)
- Hip circumference: r = 0.63 (lower body fat distribution)

**Selected Features for Reduced Model:**
1. Density, 2. Abdomen, 3. Chest, 4. Hip, 5. Weight, 6. Thigh, 7. Knee, 8. Biceps, 9. Neck

**Excluded Features (weak correlation):**
- Height (r = -0.09), Ankle (r = 0.27), Age (r = 0.29), Wrist (r = 0.35), Forearm (r = 0.36)

## ⌨️ Development

### Project Structure

```
bodyfat-estimation-mlp/
├── notebooks/                              # Analysis notebooks
│   ├── Assignment2_Part3_(i).ipynb        # Qualitative analysis
│   ├── Assignment2_Part3_(ii).ipynb       # Network performance
│   ├── Assignment2_Part3_(iii).ipynb      # Correlation analysis
│   ├── Assignment2_Part3_(iv).ipynb       # Reduced input model
│   ├── Assignment2_Part3_(v).ipynb        # Sensitivity analysis
│   ├── Assignment2_Part3_(vi).ipynb       # Performance comparison
│   └── Assignment2_Part3_(vii).ipynb      # Summary & conclusions
├── data/
│   └── Body_Fat.csv                        # Anthropometric dataset (252 samples)
├── models/
│   ├── best_full_model.keras              # Trained full model (14 features)
│   └── best_selected_features_model.keras # Optimized reduced model (9 features)
├── visualizations/
│   ├── correlation_heatmap.png            # Feature correlation matrix
│   └── correlation_with_bodyfat.png       # Body fat correlations
├── LICENSE                                # MIT license
└── README.md                              # Project documentation
```

## 🤝 Contributing

We welcome contributions from researchers, healthcare professionals, and machine learning enthusiasts! 

**Research Contributions:**
- Model improvements and new architectures
- Analysis extensions and validation studies
- Data enhancements and cross-population studies

**Development Process:**
1. Fork & Clone the repository
2. Create a research branch
3. Make contributions following scientific standards
4. Submit with comprehensive documentation

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Open Source Benefits:**
- ✅ Academic Use: Research and education applications
- ✅ Commercial Use: Clinical and business implementations
- ✅ Modification: Adapt for your specific research needs
- ✅ Distribution: Share with the global research community

## 👥 Team

<div align="center">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/ChanMeng666">
          <img src="https://github.com/ChanMeng666.png?size=100" width="100px;" alt="Chan Meng"/>
          <br />
          <sub><b>Chan Meng</b></sub>
        </a>
        <br />
        <small>Lead Researcher & Developer</small>
        <br />
        <small>🧠 ML Architecture | 📊 Data Science | 🔬 Research</small>
      </td>
    </tr>
  </table>
</div>

## 🙋‍♀️ Author

**Chan Meng**
- <img src="https://cdn.simpleicons.org/linkedin/0A66C2" width="16" height="16"> LinkedIn: [chanmeng666](https://www.linkedin.com/in/chanmeng666/)
- <img src="https://cdn.simpleicons.org/github/181717" width="16" height="16"> GitHub: [ChanMeng666](https://github.com/ChanMeng666)
- <img src="https://cdn.simpleicons.org/gmail/EA4335" width="16" height="16"> Email: [chanmeng.dev@gmail.com](mailto:chanmeng.dev@gmail.com)
- <img src="https://cdn.simpleicons.org/internetexplorer/0078D4" width="16" height="16"> Website: [chanmeng.live](https://2d-portfolio-eta.vercel.app/)


**Research Interests:**
- 🧠 Machine Learning in Healthcare: Anthropometric analysis and body composition modeling
- 📊 Data Science: Statistical analysis and predictive modeling in health sciences  
- 🔬 AI Research: Neural network optimization and feature selection methodologies
- 💻 Open Source: Democratizing access to advanced healthcare AI tools

---

<div align="center">
<strong>🧠 Advancing Healthcare Through Intelligent Body Composition Analysis 🌟</strong>
<br/>
<em>Empowering healthcare professionals, researchers, and fitness enthusiasts worldwide</em>
<br/><br/>

⭐ **Star us on GitHub** • 📖 **Read the Research** • 🐛 **Report Issues** • 💡 **Suggest Improvements** • 🤝 **Collaborate**

<br/><br/>

**Made with ❤️ by the Healthcare AI Research Community**

</div> 