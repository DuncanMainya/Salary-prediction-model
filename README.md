# Salary Prediction Model

This repository contains a machine learning model that predicts salaries based on survey data. The implementation is provided in a Jupyter notebook with comprehensive data analysis, preprocessing, and model training workflows.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Understanding](#data-understanding)
- [Installation](#installation)
- [Usage](#usage)
- [Model Saving](#model-saving)
- [Example Use Cases](#example-use-cases)

## Project Overview

This project builds a salary prediction model using machine learning techniques. The notebook demonstrates the complete pipeline from data loading to model deployment, including:

- Data exploration and visualization
- Feature engineering and preprocessing
- Model training and evaluation
- Model persistence for production use

**Use Case Example**: A recruiting company could use this model to provide salary estimates to job seekers based on their skills, experience, and location, helping them negotiate better compensation packages.

## Data Understanding

The analysis uses the Stack Overflow Developer Survey dataset (`survey_results_public.csv`), which contains comprehensive information about developers worldwide. The initial data exploration includes:

- Loading data using pandas DataFrame
- Examining dataset structure and dimensions
- Identifying missing values and data types
- Statistical summary of key variables

**Example Scenario**: When analyzing the survey data, you might discover that developers in San Francisco with 5+ years of Python experience earn significantly more than those in smaller cities, informing your feature selection strategy.

## Installation

Ensure you have Python 3.7+ installed, then install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

### Required Libraries:
- `numpy` - Numerical computing
- `pandas` - Data manipulation and analysis
- `matplotlib` - Data visualization
- `seaborn` - Statistical data visualization
- `scikit-learn` - Machine learning algorithms
- `joblib` - Model serialization

## Usage

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd salary-prediction-model
   ```

2. **Prepare your data:**
   - Download the Stack Overflow Developer Survey dataset
   - Place `survey_results_public.csv` in the project root directory

3. **Run the notebook:**
   ```bash
   jupyter notebook salary_prediction_model.ipynb
   ```
   Or with Jupyter Lab:
   ```bash
   jupyter lab salary_prediction_model.ipynb
   ```

4. **Execute the pipeline:**
   - Run all cells sequentially to complete the full analysis
   - The notebook will automatically handle data loading, preprocessing, training, and evaluation

### Example Workflow

**Scenario**: You're an HR manager wanting to predict salaries for new developer positions.

1. **Data Loading**: The notebook loads survey responses from thousands of developers
2. **Feature Engineering**: Converts categorical variables (like programming languages, education level) into numerical features
3. **Model Training**: Trains multiple algorithms and selects the best performer
4. **Prediction**: Uses the trained model to estimate salaries for new job postings

## Model Saving

The trained model is automatically saved for future use:

```python
import joblib
joblib.dump(best_model, "best_model.pkl")
```

### Loading the Saved Model

**Use Case Example**: A web application can load the saved model to provide real-time salary predictions:

```python
import joblib
import numpy as np

# Load the trained model
loaded_model = joblib.load("best_model.pkl")

# Example prediction for a Senior Python Developer in New York
# Features: [years_experience, python_skill, location_encoded, education_level]
sample_input = np.array([[8, 1, 5, 3]])
predicted_salary = loaded_model.predict(sample_input)
print(f"Predicted salary: ${predicted_salary[0]:,.2f}")
```

## Example Use Cases

### 1. Job Market Analysis
**Scenario**: A career counselor uses the model to advise students on lucrative tech skills.
- Input: Student's current skills and education
- Output: Projected salary ranges for different career paths

### 2. Recruitment Pricing
**Scenario**: A startup needs to set competitive salary offers for developer positions.
- Input: Job requirements, location, experience level
- Output: Market-competitive salary recommendations

### 3. Performance Benchmarking
**Scenario**: An existing employee wants to evaluate their current compensation.
- Input: Their skills, experience, and location
- Output: Salary prediction compared to market rates

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for improvements.

## License

This project is open source and available under the MIT License.
