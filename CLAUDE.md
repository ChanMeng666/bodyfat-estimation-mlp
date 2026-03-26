# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning research project focused on body fat estimation using Multi-Layer Perceptron (MLP) neural networks. The project analyzes anthropometric measurements to predict body fat percentage with high accuracy (R² > 0.97).

## Key Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Analysis
```bash
# Launch Jupyter Notebook environment
jupyter notebook

# Start Jupyter Lab (alternative)
jupyter lab
```

### Model Operations
```python
# Load pre-trained models
from tensorflow.keras.models import load_model
full_model = load_model('models/best_full_model.keras')
optimized_model = load_model('models/best_selected_features_model.keras')

# Load dataset
import pandas as pd
df = pd.read_csv('data/body_fat.csv')
```

## Repository Architecture

### Core Components
- **Analysis Notebooks** (`notebooks/`): Seven-part comprehensive analysis workflow
  - `01_qualitative_analysis.ipynb`: Qualitative analysis with visualization
  - `02_network_performance.ipynb`: Network performance optimization
  - `03_correlation_analysis.ipynb`: Correlation analysis
  - `04_reduced_input_model.ipynb`: Reduced input model development
  - `05_sensitivity_analysis.ipynb`: Sensitivity analysis
  - `06_performance_comparison.ipynb`: Performance comparison
  - `07_summary_and_conclusions.ipynb`: Summary and conclusions

- **Data** (`data/`): `body_fat.csv` - 252 anthropometric measurements with 14 features
- **Models** (`models/`): Pre-trained Keras models (.keras format)
  - `best_full_model.keras` - 20 neurons, all 14 features, R² = 0.9724
  - `best_selected_features_model.keras` - 5 neurons, 9 features, R² = 0.9950
- **Figures** (`figures/`): Visualization outputs (correlation heatmap, feature charts)

### Model Architecture
- **Input Features**: Age, Weight, Height, Neck, Chest, Abdomen, Hip, Thigh, Knee, Ankle, Biceps, Forearm, Wrist, Density
- **Target Variable**: BodyFat percentage
- **Neural Network**: Single hidden layer MLP with sigmoid activation
- **Optimization**: Adam optimizer with MSE loss

### Feature Selection Strategy
The optimized model uses 9 selected features based on correlation analysis:
1. Density (r = -0.99)
2. Abdomen (r = 0.81)
3. Chest (r = 0.70)
4. Hip (r = 0.63)
5. Weight (r = 0.61)
6. Thigh, Knee, Biceps, Neck

Excluded weak predictors: Height, Ankle, Age, Wrist, Forearm

### Data Processing Pipeline
1. Load CSV data with pandas
2. Feature scaling using MinMaxScaler
3. Train/test split (80/20)
4. Neural network training with early stopping
5. Model evaluation using R² score and MSE

## Working with the Codebase

### Notebook Execution Order
Run notebooks sequentially from `notebooks/` for complete analysis:
1. `01_qualitative_analysis.ipynb` — Visual exploration
2. `02_network_performance.ipynb` — Network optimization
3. `03_correlation_analysis.ipynb` — Correlation testing
4. `04_reduced_input_model.ipynb` — Feature selection
5. `05_sensitivity_analysis.ipynb` — Feature importance
6. `06_performance_comparison.ipynb` — Model comparison
7. `07_summary_and_conclusions.ipynb` — Summary

### Model Training Workflow
```python
# Standard preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model architecture
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential([
    Dense(neurons, activation='sigmoid', input_shape=(n_features,)),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse')
```

### Visualization Standards
- Use matplotlib and seaborn for consistent plotting
- Standard figure size: 10x6 inches
- Include correlation heatmaps and scatter plots
- Export key visualizations as PNG files

## Development Guidelines

### Language and Content Requirements
- **English Only**: All code comments, strings, UI text, and documentation must be in English
- **No Chinese Content**: Ensure no Chinese characters appear in any code files or components
- **Function Documentation**: Follow Google open source style guide for function-level comments

### Git Workflow and Commit Standards
- **Conventional Commits**: Use conventional commit format for all commits
- **Angular Style**: Follow Angular commit message convention in English
- **GitHub CLI**: Use `gh` command for GitHub operations (issues, PRs, releases)

```bash
# Example commit format
git commit -m "feat: add model validation pipeline"
git commit -m "fix: resolve data preprocessing edge case"
git commit -m "docs: update API documentation"
```

### Testing and Quality Assurance
- **Milestone Testing**: Implement comprehensive tests for each development milestone
- **Functional Tests**: Create targeted tests in project folder for core functionality
- **Minimal Validation**: Use smallest possible tests to verify implementation
- **Steady Progress**: Ensure robust testing enables stable development progression

### Code Architecture and Design
- **Modular Components**: Separate UI, logic, and data components for loose coupling
- **Fine-Grained Modules**: Break components into smaller, reusable units
- **Shared Logic Extraction**: Extract common functionality into utility modules
- **Code Quality**: Maintain robustness, extensibility, and maintainability

### Development Best Practices
- **Problem-Focused**: Address specific issues directly, avoid workarounds
- **Efficient Implementation**: Minimize code changes and optimize for token efficiency
- **Task-Oriented**: Focus strictly on user requirements, avoid unnecessary features
- **Clear Communication**: Express actions clearly in conversation

### Testing Commands
```bash
# Run basic functionality tests
python -m pytest tests/ -v

# Test model loading and prediction
python -c "from tensorflow.keras.models import load_model; model = load_model('models/best_full_model.keras'); print('Model loaded successfully')"

# Validate data pipeline
python -c "import pandas as pd; df = pd.read_csv('data/body_fat.csv'); print(f'Dataset loaded: {df.shape}')"
```

## Research Context

This project follows established protocols for anthropometric analysis and neural network validation in healthcare applications. The models achieve clinical-grade accuracy suitable for body composition assessment in fitness and medical applications.

The analysis methodology emphasizes feature importance, model interpretability, and validation across multiple metrics to ensure robust performance for real-world deployment.