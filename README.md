

# Titanic Survival Prediction Project

This project aims to predict the survival of passengers aboard the Titanic based on factors such as gender, passenger class, and age. The dataset, sourced from [Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic), contains information about various passenger characteristics, allowing us to explore the relationships between these factors and survival rates.

## Project Overview
The primary goal of this project is to build a machine learning model to predict whether a passenger survived the Titanic disaster. The project involved the following steps:
- Data preprocessing and exploratory data analysis (EDA)
- Feature engineering and handling missing values
- Training multiple machine learning models with hyperparameter tuning
- Evaluating model performance and selecting the best model
- Analyzing survival patterns based on gender, passenger class, and age group

## Key Insights
- **Gender**: Female passengers had a significantly higher survival rate than male passengers.
- **Passenger Class**: Passengers in first class had a higher survival rate than those in second or third class, suggesting that passenger class played a role in evacuation priority.
- **Age**: Children had a higher survival rate compared to adults and seniors, consistent with the policy of prioritizing "women and children."

## Project Structure
The project is organized as follows:

```
Titanic_Survival_Prediction/
├── data/
│   ├── TitanicTrain.csv         # Training data
│   ├── TitanicTest.csv          # Test data
├── notebooks/
│   ├── titanic_analysis.ipynb   # Jupyter Notebook with analysis and model training
├── reports/
│   ├── figures/                 # Visualizations generated from EDA and analysis
├── scripts/
│   ├── preprocessing.py         # Script for data preprocessing and feature engineering
│   ├── model_training.py        # Script for training and tuning models
├── README.md                    # Project overview and setup instructions
└── requirements.txt             # Project dependencies
```

## Installation
To set up the project environment, follow these steps:
1. **Clone the repository**:
   ```bash
   git clone 
   cd Titanic_Survival_Prediction
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. **Data Preprocessing**: Run the `preprocessing.py` script to handle missing values, encode categorical variables, and create additional features (e.g., age groups).

2. **Model Training**: Run `model_training.py` to train and evaluate multiple models, including Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting. The best model parameters and scores are saved for further analysis.

3. **Jupyter Notebook**: Use `titanic_analysis.ipynb` for a step-by-step analysis of the project. The notebook includes EDA, model training, evaluation, and visualizations.

4. **Analysis and Evaluation**: After training, the notebook and scripts will output evaluation metrics and survival distribution plots by gender, class, and age group.

## Model Performance
The best-performing models and their metrics:

| Model               | Accuracy | F1 Score |
|---------------------|----------|----------|
| Logistic Regression | 86.67%   | 83.33%   |
| Decision Tree       | 78.89%   | 71.64%   |
| Random Forest       | 81.11%   | 75.36%   |
| Gradient Boosting   | 82.22%   | 77.14%   |

The **Logistic Regression** model provided interpretability and good accuracy, while the **Gradient Boosting** model achieved the highest F1 score and accuracy among ensemble methods, capturing survival patterns effectively.

## Key Files
- `titanic_analysis.ipynb`: Jupyter Notebook with the entire analysis, model training, and evaluation.
- `preprocessing.py`: Python script for data preprocessing and feature engineering.
- `model_training.py`: Script to train, tune, and evaluate multiple models.
- `requirements.txt`: List of dependencies required to run the project.

## Visualizations
The project includes several visualizations for understanding survival distributions:
- Survival by Gender
- Survival by Passenger Class
- Survival by Age Group

These plots help highlight the influence of different factors on survival probability.

## Future Improvements
- **Ensemble Stacking**: Combine multiple models for potentially higher accuracy.
- **Advanced Feature Engineering**: Explore additional features like family size or fare groupings.
- **Hyperparameter Tuning**: Further refine models to achieve better generalization.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic) for providing the data used in this project.
- The open-source data science and machine learning libraries used for analysis and modeling.

## Contact
For any questions or collaboration opportunities, please reach out. 

--- 
