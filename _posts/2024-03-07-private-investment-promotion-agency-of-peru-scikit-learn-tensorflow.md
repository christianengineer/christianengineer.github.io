---
title: Private Investment Promotion Agency of Peru (Scikit-Learn, TensorFlow) Investment Analyst pain point is assessing project viability, solution is to use machine learning to evaluate investment projects based on economic and social impact criteria, enhancing decision-making
date: 2024-03-07
permalink: posts/private-investment-promotion-agency-of-peru-scikit-learn-tensorflow
layout: article
---

## Private Investment Promotion Agency of Peru - Machine Learning Solution

## Objectives and Benefits for Investment Analysts:
- **Objective**: Evaluate investment projects based on economic and social impact criteria.
- **Benefits**:
  - Improved decision-making process.
  - Enhanced accuracy and efficiency in assessing project viability.
  - Prioritization of investment opportunities based on data-driven insights.
  
## Machine Learning Algorithm:
- **Algorithm**: Random Forest Classifier
  - **Reasoning**: Offers high accuracy, handles non-linear data well, and provides feature importances for better interpretability.

## Sourcing, Preprocessing, Modeling, and Deploying Strategies:
1. **Sourcing Data**:
   - **Data Source**: 
      - Collect data on investment projects from various sources like reports, surveys, and databases.
   - **Tools**: Python libraries like Pandas, NumPy for data manipulation.

2. **Data Preprocessing**:
   - **Steps**:
      - Handle missing values and outliers.
      - Perform feature engineering and selection.
      - Encode categorical variables.
   - **Tools**: Scikit-learn for preprocessing tasks.

3. **Modeling**:
   - **Model**: Random Forest Classifier
   - **Steps**:
      - Split data into training and testing sets.
      - Train the model on the training set.
      - Evaluate model performance using metrics like accuracy, precision, recall.
   - **Tools**: Scikit-learn for training and evaluation.

4. **Deployment**:
   - **Strategy**: Deploy model through a REST API for easy integration with existing systems.
   - **Tools**: Flask or FastAPI for building the API. Docker for containerization.

## Tools and Libraries:
- **Python**: https://www.python.org/
- **Scikit-learn**: https://scikit-learn.org/
- **TensorFlow**: https://www.tensorflow.org/
- **Pandas**: https://pandas.pydata.org/
- **NumPy**: https://numpy.org/
- **Flask**: https://flask.palletsprojects.com/
- **FastAPI**: https://fastapi.tiangolo.com/
- **Docker**: https://www.docker.com/

By following these strategies and utilizing the mentioned tools and libraries, the Private Investment Promotion Agency of Peru can successfully implement a scalable, production-ready machine learning solution to address the Investment Analysts' pain point of evaluating investment projects.

## Sourcing Data Strategy for Investment Project Evaluation

## Data Collection Tools and Methods:
1. **Web Scraping**:
   - **Tool**: Beautiful Soup and Scrapy
     - **Description**: Extract data from websites, reports, and online databases efficiently.
     - **Integration**: Automate data extraction and store in a structured format for easy access.

2. **API Integration**:
   - **Tool**: Requests library in Python
     - **Description**: Fetch data from external APIs such as economic indicators, social impact data, and financial market information.
     - **Integration**: Integrate API calls within scripts to retrieve real-time data for analysis.

3. **Database Querying**:
   - **Tool**: SQL (e.g., SQLite, PostgreSQL)
     - **Description**: Query structured databases to retrieve relevant project data.
     - **Integration**: Connect to databases storing project information to retrieve data for evaluation.

4. **Data Partnerships**:
   - **Approach**: Collaborate with relevant organizations or data providers for access to exclusive datasets.
     - **Integration**: Establish data sharing agreements and pipelines to access required data securely.

## Integration within Existing Technology Stack:
- **Data Storage**: Utilize databases (e.g., PostgreSQL) to store collected data securely and in a structured format.
- **ETL Processes**: Implement ETL (Extract, Transform, Load) pipelines using tools like Apache Airflow to preprocess and integrate data into the system.
- **Automation**: Use cron jobs or scheduling tools to automate data collection processes at regular intervals.
- **Version Control**: Manage data collection scripts using version control systems like Git for reproducibility and collaboration.

By incorporating these tools and methods within the existing technology stack, the data collection process for the project can be streamlined, ensuring that the data is readily accessible, up-to-date, and in the correct format for analysis and model training. This approach will facilitate efficient evaluation of investment projects based on economic and social impact criteria, leading to improved decision-making for Investment Analysts at the Private Investment Promotion Agency of Peru.

## Feature Extraction and Engineering Strategy for Investment Project Evaluation

## Feature Extraction:
1. **Economic Indicators**:
   - *GDP Growth Rate*: Measure of economic growth
   - *Interest Rates*: Impact on investment decisions
   - *Inflation Rate*: Reflects purchasing power
2. **Social Impact Data**:
   - *Employment Rate*: Indicates labor market conditions
   - *Poverty Rate*: Measure of social inequality
   - *Education Level*: Influence on workforce productivity

## Feature Engineering:
1. **Time-Related Features**:
   - *Quarter*: Extract quarter from project timeline
   - *Year*: Extract year from project timeline
2. **Categorical Variables**:
   - *Region*: Encode geographical location of the project
   - *Sector*: Categorize the industry sector of the project
3. **Interaction Features**:
   - *GDP Growth Rate x Sector*: Interaction between economic performance and sector
   - *Employment Rate x Region*: Impact of regional employment on project viability

## Variable Naming Recommendations:
1. **Economic Indicators**:
   - *gdp_growth_rate*
   - *interest_rate*
   - *inflation_rate*
2. **Social Impact Data**:
   - *employment_rate*
   - *poverty_rate*
   - *education_level*
3. **Time-Related Features**:
   - *quarter*
   - *year*
4. **Categorical Variables**:
   - *region_encoded*
   - *sector_encoded*
5. **Interaction Features**:
   - *gdp_sector_interaction*
   - *employment_region_interaction*

By incorporating these feature extraction and engineering strategies, the project can enhance both the interpretability of the data and the performance of the machine learning model. The recommended variable names follow a consistent and descriptive naming convention, aiding in better understanding and analysis of the dataset. This approach will optimize the development and effectiveness of the project's objectives, ultimately improving the evaluation of investment projects for Investment Analysts at the Private Investment Promotion Agency of Peru.

## Metadata Management Recommendations for Investment Project Evaluation

1. **Variable Descriptions**:
   - **Economic Indicators**:
     - *gdp_growth_rate*: Percentage change in GDP over a period
     - *interest_rate*: Rate at which interest is charged on borrowed funds
     - *inflation_rate*: Rate of price increase of goods and services
   - **Social Impact Data**:
     - *employment_rate*: Percentage of the workforce currently employed
     - *poverty_rate*: Percentage of the population living below the poverty line
     - *education_level*: Level of education attained by the workforce

2. **Variable Types**:
   - **Continuous Variables**:
     - *gdp_growth_rate*, *interest_rate*, *inflation_rate*
   - **Discrete Variables**:
     - *employment_rate*, *poverty_rate*, *education_level*
   - **Categorical Variables**:
     - *region_encoded*, *sector_encoded*

3. **Variable Relationships**:
   - **Correlations**:
     - Explore correlations between economic indicators and social impact data
   - **Interactions**:
     - Analyze interaction effects between different variables (e.g., GDP growth rate and sector)

4. **Missing Data Handling**:
   - **Imputation**:
     - Define strategies for imputing missing values in variables like *employment_rate* and *education_level*

5. **Outlier Detection**:
   - **Identification**:
     - Establish thresholds for outlier detection in economic and social impact variables
   - **Treatment**:
     - Determine whether to remove, transform, or impute outliers based on their impact on the analysis

By managing metadata specific to the characteristics and demands of the investment project evaluation, the Private Investment Promotion Agency of Peru can ensure that the data used for model training and analysis is well-understood, accurately represented, and aligned with the unique requirements of the project. Taking into account these insights will enhance the effectiveness and success of the machine learning solution in evaluating investment projects based on economic and social impact criteria.

## Data Challenges and Preprocessing Strategies for Investment Project Evaluation

## Specific Data Problems:
1. **Missing Values**:
   - Economic and social impact data may have missing values, impacting analysis and model performance.
2. **Outliers**:
   - Outliers in economic indicators or social impact variables can skew results and model predictions.
3. **Data Imbalance**:
   - Disproportionate distribution of project data across regions or sectors can lead to biased model outcomes.
4. **Feature Scaling**:
   - Economic indicators and social impact data may have different scales, affecting model convergence and performance.
5. **Categorical Variables**:
   - Encoding categorical variables like region and sector requires careful handling to avoid bias and ensure model interpretability.

## Data Preprocessing Strategies:
1. **Handling Missing Values**:
   - Impute missing values using appropriate methods like mean, median, or advanced imputation techniques based on the variable characteristics.
2. **Outlier Detection and Treatment**:
   - Identify and handle outliers through techniques such as trimming, Winsorization, or transformation to mitigate their impact on model training.
3. **Addressing Data Imbalance**:
   - Employ techniques like oversampling, undersampling, or using ensemble methods to balance the distribution of data across regions and sectors.
4. **Feature Scaling**:
   - Normalize or standardize numerical features to bring them to a similar scale and prevent dominance of certain features in the model.
5. **Categorical Variable Encoding**:
   - Utilize techniques like one-hot encoding or ordinal encoding to represent categorical variables numerically while preserving the inherent relationships.

By strategically employing these data preprocessing practices tailored to the specific challenges of the investment project evaluation, the data can be cleansed, standardized, and optimized for model training. This ensures that the data remains robust, reliable, and conducive to developing high-performing machine learning models that accurately evaluate investment projects based on economic and social impact criteria.

```python
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

## Load the dataset
data = pd.read_csv('investment_project_data.csv')

## Define features and target variable
X = data.drop('target_variable', axis=1)
y = data['target_variable']

## Preprocessing steps
numeric_features = ['gdp_growth_rate', 'interest_rate', 'inflation_rate', 'employment_rate', 'poverty_rate']
categorical_features = ['region_encoded', 'sector_encoded']

## Impute missing values in numeric features with median
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())  ## Standardize numeric features
])

## One-hot encode categorical features
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])

## Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

## Fit preprocessor on the data
X_preprocessed = preprocessor.fit_transform(X)

## Check the shape of preprocessed data
print(X_preprocessed.shape)
```

In the provided code:
- The data is loaded and divided into features (X) and the target variable (y).
- Numeric features are standardized with missing values imputed using the median to handle missing data effectively.
- Categorical features are one-hot encoded to convert them into numerical format for model training without introducing ordinal relationships.
- The `preprocessor` pipeline combines the numeric and categorical feature transformations for seamless preprocessing.
- The preprocessor is fitted on the data to transform it into a format ready for model training, ensuring consistency and compatibility with machine learning algorithms.

These preprocessing steps are tailored to handle specific challenges of the investment project data such as missing values, disparate scales, and categorical variables encoding, preparing the data optimally for effective model training and analysis in evaluating investment projects for the Private Investment Promotion Agency of Peru.

## Modeling Strategy for Investment Project Evaluation

## Recommended Modeling Strategy:
- **Algorithm**: Gradient Boosting Classifier
  - **Reasoning**: 
    - Handles complex relationships well and provides high accuracy.
    - Robust to overfitting and works with a mix of data types.
  
- **Hyperparameter Tuning**:
  - Utilize techniques like grid search or random search to optimize the model's hyperparameters for enhanced performance and generalization.

- **Cross-Validation**:
  - Implement k-fold cross-validation to assess the model's performance robustly on different subsets of data and mitigate overfitting.

- **Evaluation Metrics**:
  - Focus on metrics such as F1 Score and Area Under the ROC Curve (AUC-ROC) to evaluate the model's ability to balance precision and recall, crucial for investment project decision-making accuracy.

- **Feature Importance Analysis**:
  - Conduct feature importance analysis to identify the key economic and social impact factors influencing investment project viability, aiding in decision-making interpretation.

## Most Crucial Step: Hyperparameter Tuning
The most critical step in this modeling strategy is hyperparameter tuning. In the context of evaluating investment projects based on economic and social impact criteria, hyperparameter tuning is vital because it fine-tunes the model's parameters to optimize performance and generalization. Since the project deals with diverse data types and complex relationships between features, tuning the algorithm's hyperparameters ensures that the model can effectively capture these nuances and make accurate predictions. This step is crucial for enhancing the model's predictive power and ensuring that it can effectively evaluate investment projects, ultimately improving decision-making for Investment Analysts at the Private Investment Promotion Agency of Peru.

## Tools and Technologies Recommendations for Data Modeling in Investment Project Evaluation:

1. **Tool: XGBoost**
   - **Description**: XGBoost is an optimized distributed gradient boosting library known for its efficiency, speed, and performance in handling diverse data types.
   - **Fit with Modeling Strategy**: XGBoost is well-suited for handling the complexities of the project's data, such as mixed data types and complex relationships, aligning with the Gradient Boosting Classifier modeling strategy.
   - **Integration**: XGBoost can easily integrate with Python and Scikit-learn, ensuring seamless inclusion in the existing workflow.
   - **Beneficial Features**: Rich set of hyperparameters for tuning, efficient handling of missing data, and feature importance analysis.
   - **Documentation**: [XGBoost Documentation](https://xgboost.readthedocs.io/)

2. **Tool: Hyperopt**
   - **Description**: Hyperopt is a Python library for optimizing model hyperparameters using Bayesian optimization techniques.
   - **Fit with Modeling Strategy**: Hyperopt is crucial for hyperparameter tuning, optimizing the Gradient Boosting Classifier for improved performance and generalization.
   - **Integration**: Compatible with popular machine learning libraries like Scikit-learn, providing seamless integration into the modeling workflow.
   - **Beneficial Features**: Bayesian optimization for efficient parameter search, customizable search spaces, and parallel processing capabilities.
   - **Documentation**: [Hyperopt Documentation](https://github.com/hyperopt/hyperopt)

3. **Tool: SHAP (SHapley Additive exPlanations)**
   - **Description**: SHAP is a Python library for interpreting machine learning models, providing insights into feature importance and model predictions.
   - **Fit with Modeling Strategy**: SHAP is essential for conducting feature importance analysis, aiding in understanding the key factors influencing investment project viability.
   - **Integration**: Compatible with a wide range of machine learning frameworks, enabling seamless integration for interpreting model decisions.
   - **Beneficial Features**: Unified framework for feature importance, model explanation, and visualization.
   - **Documentation**: [SHAP Documentation](https://github.com/slundberg/shap)

By leveraging these recommended tools and technologies tailored to the specific data modeling needs of the investment project evaluation, the Private Investment Promotion Agency of Peru can enhance efficiency, accuracy, and scalability in assessing investment projects based on economic and social impact criteria. The seamless integration of these tools into the existing workflow will enable Investment Analysts to make informed decisions and drive successful project evaluations.

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

## Generate a large fictitious dataset
n_samples = 10000

## Create dummy data for economic indicators
data = {
    'gdp_growth_rate': np.random.uniform(low=0, high=10, size=n_samples),
    'interest_rate': np.random.uniform(low=1, high=5, size=n_samples),
    'inflation_rate': np.random.uniform(low=0, high=3, size=n_samples),
    'employment_rate': np.random.uniform(low=30, high=70, size=n_samples),
    'poverty_rate': np.random.uniform(low=5, high=50, size=n_samples),
    'education_level': np.random.randint(low=1, high=5, size=n_samples),
    'region_encoded': np.random.choice(['Lima', 'Arequipa', 'Cusco', 'Ica'], size=n_samples),
    'sector_encoded': np.random.choice(['Agriculture', 'Manufacturing', 'Services'], size=n_samples)
}

df = pd.DataFrame(data)

## One-hot encode categorical variables
encoder = OneHotEncoder()
encoded_categorical = encoder.fit_transform(df[['region_encoded', 'sector_encoded']]).toarray()
encoded_df = pd.concat([df.drop(['region_encoded', 'sector_encoded'], axis=1), pd.DataFrame(encoded_categorical)], axis=1)

## Add noise to simulate real-world variability
for col in encoded_df.columns:
    if encoded_df[col].dtype != 'object':
        encoded_df[col] += np.random.normal(0, 1, n_samples)

## Save dataset to CSV
encoded_df.to_csv('fictional_investment_data.csv', index=False)

## Validate dataset generation
print("Dataset created and saved successfully.")
```

In the provided Python script:
- A large fictitious dataset is generated to mimic real-world data relevant to the investment project evaluation.
- The script includes attributes for economic indicators, social impact data, region encoded, and sector encoded as per the project's feature requirements.
- Categorical variables are one-hot encoded to represent them numerically for model training.
- Noise is added to simulate real-world variability in the dataset.
- The final dataset is saved as a CSV file for use in model testing and validation.

By using this script with the recommended tools for dataset creation and validation, Investment Analysts at the Private Investment Promotion Agency of Peru can generate a dataset that accurately simulates real conditions, incorporating variability and ensuring compatibility with the modeling strategies. This will enhance the model's predictive accuracy and reliability in evaluating investment projects based on economic and social impact criteria.

```plaintext
Sample Mocked Dataset for Investment Project Evaluation:

| gdp_growth_rate | interest_rate | inflation_rate | employment_rate | poverty_rate | education_level | region_Lima | region_Arequipa | region_Cusco | region_Ica | sector_Agriculture | sector_Manufacturing | sector_Services |
|-----------------|---------------|----------------|-----------------|--------------|-----------------|-------------|-----------------|-------------|-----------|-------------------|---------------------|-----------------|
| 5.2             | 3.8           | 1.2            | 45.6            | 15.3         | 3               | 1           | 0               | 0           | 0         | 1                 | 0                   | 0               |
| 7.8             | 2.2           | 0.5            | 62.3            | 8.9          | 2               | 0           | 1               | 0           | 0         | 0                 | 1                   | 0               |
| 3.6             | 4.5           | 2.8            | 38.9            | 25.6         | 4               | 0           | 0               | 1           | 0         | 0                 | 0                   | 1               |

Key:
- Numeric Features: gdp_growth_rate, interest_rate, inflation_rate, employment_rate, poverty_rate
- Categorical Features: region_encoded (Lima, Arequipa, Cusco, Ica), sector_encoded (Agriculture, Manufacturing, Services)
- One-Hot Encoded Representation: 1 indicates presence, 0 indicates absence

Note: This sample dataset showcases a few rows of data with relevant features structured in a format suitable for model ingestion for the investment project evaluation.
```

In the provided sample mocked dataset representation:
- Rows of data showcase the values for economic indicators, social impact data, region encoded, and sector encoded as relevant to the investment project evaluation.
- Features are structured with clear names and types, including numeric features and one-hot encoded categorical features for regions and sectors.
- The one-hot encoded representation highlights the presence or absence of each category, facilitating model ingestion and interpretation for the project's objectives.

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

## Load preprocessed dataset
data = pd.read_csv('preprocessed_data.csv')

## Define features and target variable
X = data.drop('target_variable', axis=1)
y = data['target_variable']

## Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Initialize Gradient Boosting Classifier
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

## Train the model
model.fit(X_train, y_train)

## Make predictions on the test set
y_pred = model.predict(X_test)

## Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

## Save the model for deployment
import joblib
joblib.dump(model, 'investment_project_model.pkl')
```

In the provided Python code snippet:
- The code is structured for immediate deployment in a production environment, focusing on the model's data for the investment project evaluation.
- Comments are included to explain the logic, purpose, and functionality of key sections, adhering to best practices for documentation.
- Conventions for code quality and structure in large tech environments are followed, including clear variable names, modularization of code, and concise yet informative comments.
- The code trains a Gradient Boosting Classifier on preprocessed data, evaluates its accuracy, and saves the model for deployment using Joblib.

By following these standards for quality, readability, and maintainability in the codebase, Investment Analysts at the Private Investment Promotion Agency of Peru can ensure the development of a robust and scalable production-level machine learning model for evaluating investment projects based on economic and social impact criteria.

## Deployment Plan for Machine Learning Model in Investment Project Evaluation

## Pre-Deployment Checks:
1. **Model Evaluation**:
   - Ensure the model meets performance criteria through thorough testing.
2. **Security and Compliance**:
   - Verify data privacy measures and compliance with regulations.
3. **Scalability Assessment**:
   - Check scalability of the model for production-level workload.

## Deployment Steps:
1. **Model Serialization**:
   - Serialize the trained model for easy deployment.
   - **Tools**: Joblib or Pickle
     - [Joblib Documentation](https://joblib.readthedocs.io/en/latest/)
2. **Containerization**:
   - Containerize the model and its dependencies for portability.
   - **Tools**: Docker
     - [Docker Documentation](https://docs.docker.com/)
3. **Model API Development**:
   - Create an API to serve model predictions.
   - **Tools**: Flask or FastAPI
     - [Flask Documentation](https://flask.palletsprojects.com/)
     - [FastAPI Documentation](https://fastapi.tiangolo.com/)
4. **Scalable Deployment**:
   - Deploy the model API on a scalable cloud platform.
   - **Tools**: AWS, Google Cloud, or Azure
     - [AWS Documentation](https://docs.aws.amazon.com/)
     - [Google Cloud Documentation](https://cloud.google.com/docs)
     - [Azure Documentation](https://docs.microsoft.com/en-us/azure/)

## Live Environment Integration:
1. **Monitoring and Logging**:
   - Implement monitoring tools for performance tracking.
   - **Tools**: Prometheus, Grafana
     - [Prometheus Documentation](https://prometheus.io/docs/)
     - [Grafana Documentation](https://grafana.com/docs/)
2. **Continuous Integration/Continuous Deployment (CI/CD)**:
   - Automate deployment pipelines for seamless updates.
   - **Tools**: Jenkins, GitHub Actions
     - [Jenkins Documentation](https://www.jenkins.io/doc/)
     - [GitHub Actions Documentation](https://docs.github.com/en/actions)

By following this deployment plan tailored to the unique demands of the investment project evaluation, the team at the Private Investment Promotion Agency of Peru can ensure a smooth and efficient transition of the machine learning model into a production environment. Each step is designed to empower the team with the necessary tools and resources to execute the deployment independently, enabling the model to be integrated seamlessly into the live environment for enhancing investment project evaluations.

```dockerfile
## Use a base image with Python and required libraries
FROM python:3.8-slim

## Set working directory in the container
WORKDIR /app

## Copy the requirements file into the container
COPY requirements.txt .

## Install necessary libraries
RUN pip install --no-cache-dir -r requirements.txt

## Copy the preprocessed data and trained model into the container
COPY preprocessed_data.csv .
COPY investment_project_model.pkl .

## Copy the Python script for serving the model predictions
COPY predict.py .

## Expose the API port
EXPOSE 5000

## Command to run the API for serving model predictions
CMD ["python", "predict.py"]
```

In the provided Dockerfile:
- The base image is set to Python with a specific version to match the project's requirements.
- Libraries are installed from the `requirements.txt` file to ensure the necessary dependencies are included in the container.
- Important project files including preprocessed data, trained model, and prediction script are copied into the container.
- The API port is exposed for communication with the outside environment.
- The command `CMD ["python", "predict.py"]` is set to run the API script for serving model predictions.

This Dockerfile is tailored to encapsulate the project's environment and dependencies, optimized for performance and scalability to meet the specific needs of the investment project evaluation in a production environment.

## User Groups and User Stories for the Investment Project Evaluation Application:

### User Group: Investment Analysts
**User Story**:
- *Scenario*: Maria, an Investment Analyst at the agency, is overwhelmed by the volume of investment projects to assess. She struggles to evaluate projects based on multiple economic and social impact criteria efficiently.
- *Application Solution*: The machine learning model in the application automates project evaluation based on economic and social impact factors. Maria can quickly obtain insights on project viability, prioritizing high-impact opportunities.
- *Facilitating Component*: The model training script and model deployment API streamline project evaluation, saving time and improving decision-making for Investment Analysts.

### User Group: Decision-Makers at the Agency
**User Story**:
- *Scenario*: Alejandro, a Decision-Maker, needs to allocate resources effectively to drive economic growth. He lacks a systematic approach to assess the potential impact of investment projects on the local economy.
- *Application Solution*: The machine learning model provides data-driven insights on the economic and social impact of investment projects. Alejandro can make informed decisions, maximizing the agency's contributions to economic development.
- *Facilitating Component*: The model's feature importance analysis helps Decision-Makers identify critical factors influencing project viability, enhancing resource allocation strategies.

### User Group: Project Managers and Stakeholders
**User Story**:
- *Scenario*: Javier, a Project Manager, struggles to communicate the potential benefits of his investment project effectively to stakeholders. He needs a comprehensive analysis to showcase the project's value proposition.
- *Application Solution*: The machine learning model evaluates projects based on economic and social impact criteria, quantifying their potential benefits. Javier can present data-backed insights to stakeholders, garnering support and funding for projects.
- *Facilitating Component*: The model's prediction capabilities and API integration enable Project Managers to generate detailed analyses for stakeholders, fostering collaboration and project success.

### User Group: Researchers and Academics
**User Story**:
- *Scenario*: Sofia, a researcher specializing in economic development, seeks data-driven insights on the impact of investments in specific regions and sectors. She requires robust analytical tools to support her research.
- *Application Solution*: The machine learning model offers sophisticated analysis of investment projects, providing valuable insights for research on economic and social impact. Sofia can leverage the model's results to enhance her studies and publications.
- *Facilitating Component*: Access to the model's predictions and feature importance analysis supports researchers in analyzing and interpreting investment project data effectively.

By identifying diverse user groups and crafting user stories that illustrate how the application addresses their pain points, the value proposition of the investment project evaluation application becomes clear. Each user type benefits from the machine learning solution by streamlining project assessments, enhancing decision-making, and providing valuable insights tailored to their specific needs within the Private Investment Promotion Agency of Peru.