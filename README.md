# Body Fat Estimation Using Neural Networks

This repository contains the code and analysis for estimating body fat percentage using feed-forward neural networks. The project is a case study for predicting body fat based on various physical measurements. The task was completed as part of the COMP 627: Neural Networks and Applications course. This project demonstrates my expertise in building and optimizing neural networks for real-world health-related datasets.

## Overview

The objective of this project was to develop a multi-layer perceptron (MLP) model to estimate body fat percentage using physical measurements like body density, age, weight, and several circumference measurements (abdomen, chest, hip, etc.). Body fat percentage is a critical indicator of health, but direct measurement methods are costly and inconvenient. Therefore, the goal was to build a model that could estimate body fat based on easily measurable attributes.

### Key Features:
- **Exploratory Data Analysis**: Conducted qualitative and quantitative analysis of input features through visualization (scatter plots, heatmaps) and statistical correlation (correlation matrices).
- **Feed-Forward Neural Network**: Developed MLP models with varying hidden layer sizes (5, 10, and 20 neurons), using Sigmoid activation functions and Adam optimization.
- **Performance Evaluation**: Compared models based on Mean Squared Error (MSE) and R² scores for training, validation, and test sets to select the best performing network.
- **Feature Selection**: Identified the most influential attributes (e.g., abdomen, chest, hip circumference) through correlation analysis and built a reduced-input model for a more efficient network with fewer measurements.

## Repository Structure

- `Body_Fat_Estimation.ipynb`: Jupyter notebook containing the Python code used for the model development, data analysis, and visualizations.
- `data/`: Contains the dataset used for training the models (`Body_Fat.csv`).
- `results/`: Includes plots and performance metrics from various model configurations.
- `Assignment_Report.pdf`: A detailed report summarizing the methodology, results, and insights from the neural network experiments.

## Key Insights

1. **Correlation Analysis**: Abdomen circumference was identified as the most influential predictor of body fat percentage, with a correlation of 0.81, followed by chest and hip circumferences.
   
2. **Neural Network Models**: 
   - The best-performing model with all input features used 20 neurons in the hidden layer, achieving an R² score of 0.9941 on the test set and an MSE of 0.2723.
   - The reduced model, using only the most significant features, achieved similar performance with fewer neurons and a simpler architecture, making it a cost-effective alternative.

3. **Cost-Effective Model**: A smaller input model with only 9 key attributes and 5 neurons in the hidden layer performed exceptionally well, reducing data collection costs while maintaining high prediction accuracy (R² = 0.9950, MSE = 0.2340).

## Skills Demonstrated

- **Neural Network Design**: Proficient in developing and tuning MLP models, including feature selection and architecture optimization.
- **Data Analysis and Visualization**: Applied advanced data exploration techniques to uncover insights and relationships between features.
- **Model Evaluation**: Evaluated model performance using MSE and R² metrics, and fine-tuned hyperparameters for optimal performance.
- **Python Programming**: Used Python libraries such as `TensorFlow`, `Keras`, `Pandas`, `NumPy`, and `Seaborn` for building neural networks, data processing, and visualization.

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/body-fat-estimation.git

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter notebook:
   ```bash
   jupyter notebook Body_Fat_Estimation.ipynb
   ```

## Conclusion

This project showcases my ability to develop effective neural network models for health-related applications, optimize model performance, and provide insights on feature importance for real-world datasets. The reduced-input model highlights the potential for a cost-effective and efficient solution in body fat estimation.
