<div align="center">
 <h1>Body Fat Prediction using Neural Networks</h1>
 <img src="https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=flat&logo=tensorflow&logoColor=white"/>
 <img src="https://img.shields.io/badge/Python-3.7+-3776AB?style=flat&logo=python&logoColor=white"/>
 <img src="https://img.shields.io/badge/Keras-2.0+-D00000?style=flat&logo=keras&logoColor=white"/>
 <img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white"/>
 <img src="https://img.shields.io/badge/License-MIT-green?style=flat"/>
</div>
<br/>

This project implements advanced neural network models for accurate prediction of body fat percentage using anthropometric measurements. Through comprehensive analysis and optimization, we've developed both full-feature and reduced-input models that achieve high accuracy while maintaining practical applicability.

# Features

### 🧠 Advanced Neural Architecture
- Multi-layer perceptron models with optimized hidden layer configurations
- Support for both comprehensive and reduced-input feature sets
- Adaptive learning with early stopping and optimization

### 📊 Comprehensive Analysis Suite
- Detailed correlation analysis of body measurements
- Feature importance evaluation through sensitivity analysis
- Extensive model performance comparisons

### 🎯 High Prediction Accuracy
- R² score of 0.9724 for full-feature model
- MSE as low as 1.9250 on test data
- Robust performance across different body types

### 💡 Smart Feature Selection
- Intelligent reduction of input measurements
- Maintains high accuracy with fewer required measurements
- Practical implementation considerations

### 📈 Performance Visualization
- Detailed performance metrics and comparisons
- Feature correlation heatmaps
- Model sensitivity analysis plots

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/ChanMeng666/bodyfat-estimation-mlp.git
cd bodyfat-estimation-mlp
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter notebooks:
```bash
jupyter notebook
```

## Model Performance

| Model Type    | R² Score | MSE    | Hidden Layers |
| ------------- | -------- | ------ | ------------- |
| Full Input    | 0.9724   | 1.9250 | 20            |
| Reduced Input | 0.9617   | 2.6734 | 5             |

## Tech Stack
![Python](https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-%23F37626.svg?style=for-the-badge&logo=Jupyter&logoColor=white)

## Project Structure
```
├── notebooks/
│   ├── Part3_(i)_Qualitative_Analysis.ipynb
│   ├── Part3_(ii)_Network_Performance.ipynb
│   ├── Part3_(iii)_Correlation_Analysis.ipynb
│   ├── Part3_(iv)_Reduced_Input_Model.ipynb
│   ├── Part3_(v)_Sensitivity_Analysis.ipynb
│   ├── Part3_(vi)_Performance_Comparison.ipynb
│   └── Part3_(vii)_Summary.ipynb
├── data/
│   └── Body_Fat.csv
├── models/
│   ├── best_full_model.keras
│   └── best_reduced_model.keras
├── requirements.txt
└── README.md
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset sourced from clinical body composition measurements
- Research methodology based on neural network optimization techniques
- Performance metrics and analysis methods from established machine learning practices

For detailed implementation and analysis, please refer to the individual notebooks in the repository.
