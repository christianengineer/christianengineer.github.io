---
title: Fishery Stock Assessment Model with Keras and TensorFlow for Pesquera Exalmar (Lima, Peru), Fisheries Manager's pain point is sustaining fish populations while maximizing catch, solution is to deploy ML models that analyze oceanographic data to predict fish stock levels, ensuring sustainable fishing practices
date: 2024-03-06
permalink: posts/fishery-stock-assessment-model-with-keras-and-tensorflow-for-pesquera-exalmar-lima-peru
---

# Objective and Benefits
## Objective:
The objective of this project is to deploy a scalable, production-ready machine learning solution using Keras and TensorFlow for Pesquera Exalmar. The solution will provide accurate predictions of fish stock levels based on oceanographic data to help ensure sustainable fishing practices.

## Benefits to the Audience:
- **Improved Decision Making:** By providing accurate predictions of fish stock levels, the fisheries manager can make informed decisions on fishing quotas and practices to sustain fish populations.
- **Optimized Catch:** The model helps in maximizing catch while ensuring the sustainability of fish populations, leading to increased profitability for Pesquera Exalmar.
- **Efficient Resource Allocation:** By predicting fish stock levels, resources can be allocated more efficiently, reducing waste and optimizing operations.

# Machine Learning Algorithm
For this specific project, we will use a **Long Short-Term Memory (LSTM)** network, a type of Recurrent Neural Network (RNN) that is well-suited for sequence prediction tasks, such as time-series forecasting. LSTMs are capable of learning long-term dependencies in sequential data, making them ideal for modeling oceanographic data and predicting fish stock levels over time.

# Sourcing, Preprocessing, Modeling, and Deploying Strategies
## Sourcing Data:
- **Oceanographic Data:** Gather historical oceanographic data such as sea surface temperature, chlorophyll levels, salinity, and other relevant features that influence fish populations.

## Preprocessing Data:
- **Feature Engineering:** Create relevant features such as seasonal trends, lagged variables, and rolling statistics to capture important patterns in the data.
- **Normalization:** Scale the features to a similar range to aid model convergence.
- **Sequence Generation:** Create input sequences and target sequences for training the LSTM model.

## Modeling:
- **LSTM Model:** Build a LSTM neural network architecture using Keras with TensorFlow backend.
- **Model Tuning:** Optimize hyperparameters, such as the number of LSTM layers, number of units in each layer, batch size, and learning rate, to improve model performance.
- **Validation:** Evaluate the model using techniques like time-series cross-validation to ensure its robustness and generalization capabilities.

## Deployment:
- **Scalability:** Deploy the model using scalable cloud services, such as AWS or Google Cloud Platform, to handle varying workloads.
- **API Integration:** Expose the model through an API endpoint for easy integration into existing systems.
- **Monitoring:** Implement monitoring tools to track model performance and detect any drift in data or model degradation.

# Tools and Libraries
- **Keras:** Deep learning library for building neural networks.
- **TensorFlow:** Deep learning framework for training and deploying ML models.
- **Pandas:** Data manipulation library for preprocessing and analysis.
- **NumPy:** Library for numerical operations and array manipulation.
- **Scikit-learn:** Library for machine learning tools and algorithms.
- **AWS/GCP:** Cloud platforms for deploying and scaling machine learning models.

*Note: Ensure to consult the official documentation of each tool and library for detailed usage instructions.*

# Sourcing Data Strategy
## Data Collection:
To efficiently collect oceanographic data for the Fishery Stock Assessment Model, we can leverage the following tools and methods tailored to cover all relevant aspects of the problem domain:

1. **API Integration:**
   - **NOAA API:** The National Oceanic and Atmospheric Administration (NOAA) provides APIs to access a wide range of oceanographic data, including sea surface temperature, chlorophyll levels, and more. Integrate the NOAA API to fetch real-time and historical oceanographic data relevant to fish populations.

2. **Satellite Imagery:**
   - **Sentinel Hub:** Utilize Sentinel Hub's satellite imagery API to retrieve high-resolution satellite images capturing sea surface temperature, chlorophyll concentrations, and other important oceanographic features. Process the satellite imagery to extract relevant data for analysis.

3. **Data Providers and Partnerships:**
   - **Oceanographic Research Institutions:** Collaborate with oceanographic research institutions and universities that collect and analyze oceanographic data. Establish partnerships to access their data repositories and research findings related to fish stock assessment.

4. **Sensor Networks:**
   - **IoT Sensor Networks:** Deploy IoT sensor networks in key fishing areas to gather real-time data on water temperature, salinity, and other environmental factors impacting fish populations. Use IoT platforms like Thingspeak or AWS IoT to collect and manage sensor data efficiently.

## Integration within Existing Technology Stack:
To streamline the data collection process and ensure that data is readily accessible and in the correct format for analysis and model training, we can integrate the data sourcing tools within Pesquera Exalmar's existing technology stack:

1. **Data Pipeline Automation:**
   - **Apache NiFi:** Implement Apache NiFi for data ingestion, processing, and routing. Create data pipelines that automatically fetch oceanographic data from APIs, satellite imagery sources, and sensor networks, transforming and storing it in a centralized data lake.

2. **Data Storage and Management:**
   - **Amazon S3:** Store raw and processed oceanographic data in Amazon S3 buckets for secure and scalable storage. Organize the data using a hierarchical structure to facilitate retrieval and analysis.

3. **Data Preprocessing and Integration:**
   - **Apache Spark:** Utilize Apache Spark for large-scale data preprocessing tasks. Clean, transform, and aggregate oceanographic data to generate features relevant to fish stock assessment, ensuring data consistency and quality.

4. **Version Control and Collaboration:**
   - **Git/GitHub:** Implement version control using Git/GitHub to track changes in data collection scripts, preprocessing pipelines, and model training code. Facilitate collaboration among data scientists and engineers working on the project.

By integrating these tools and methods within Pesquera Exalmar's technology stack, we can establish a robust data collection process that efficiently gathers oceanographic data, prepares it for analysis, and ensures seamless integration with the machine learning model training pipeline.

# Feature Extraction and Engineering Analysis

## Feature Extraction:
To enhance the interpretability of the data and the performance of the machine learning model for fish stock assessment, we propose extracting the following features from the oceanographic data:

1. **Seasonal Trends:**
   - **Feature Name:** `seasonal_trend`
   - *Description:* Extract seasonal patterns from oceanographic data to capture seasonal variations in environmental factors affecting fish populations.

2. **Lagged Variables:**
   - **Feature Name:** `lagged_sea_surface_temp`, `lagged_chlorophyll`
   - *Description:* Include lagged values of sea surface temperature and chlorophyll levels to account for the temporal dependencies in the data and capture delayed effects on fish stocks.

3. **Rolling Statistics:**
   - **Feature Name:** `rolling_mean_salinity`, `rolling_std_oxygen`
   - *Description:* Calculate rolling mean and standard deviation of salinity and oxygen levels to capture short-term trends and fluctuations that may impact fish populations.

4. **Time of Day:**
   - **Feature Name:** `time_of_day`
   - *Description:* Encode the time of day (morning, afternoon, evening) to account for diurnal variations in environmental conditions and fish behavior.

## Feature Engineering:
In addition to extracting features, we recommend performing the following feature engineering techniques to enrich the data representation and improve the model's performance:

1. **Normalization:**
   - **Normalization Method:** Min-Max Scaling
   - *Description:* Scale numerical features to a common range (e.g., [0, 1]) to prevent bias towards variables with larger magnitudes and aid model convergence.

2. **One-Hot Encoding:**
   - **Categorical Variable:** `time_of_day`
   - *Description:* Convert categorical variables like time of day into binary vectors to represent different categories independently.

3. **Interaction Features:**
   - *Description:* Create interaction features by combining relevant variables to capture synergistic effects (e.g., sea surface temperature * chlorophyll concentration).

4. **Polynomial Features:**
   - *Description:* Generate polynomial features to capture non-linear relationships between variables and enhance the model's capacity to capture complex patterns.

## Variable Names Recommendations:
To maintain clarity and consistency in variable names, we suggest the following naming conventions for the extracted and engineered features:

- **Extracted Features:**
   - Prefix: `extracted_` + feature name (e.g., `extracted_seasonal_trend`)
- **Engineered Features:**
   - Prefix: `engineered_` + feature name (e.g., `engineered_interaction_sea_surface_temp_chlorophyll`)

By following these recommendations for feature extraction, feature engineering, and variable naming conventions, we can enhance the interpretability of the data, enrich the model's input representation, and improve the overall performance of the Fishery Stock Assessment Model for Pesquera Exalmar.

# Metadata Management for Fishery Stock Assessment Model

In the context of the Fishery Stock Assessment Model for Pesquera Exalmar, effective metadata management is crucial to ensure the success of the project and address its unique demands and characteristics:

## 1. **Data Contextualization:**
- **Fish Species Metadata:** Include metadata related to the specific fish species being assessed, such as biological information, habitat preferences, and historical catch data.
- **Oceanographic Data Sources Metadata:** Document metadata for each oceanographic data source, detailing parameters measured, sensors used, and data collection methods.

## 2. **Feature Descriptions:**
- **Feature Transformation Metadata:** Document the transformation methods applied to each feature during preprocessing, including normalization, standardization, encoding, and scaling.
- **Feature Engineering Metadata:** Record the rationale behind creating engineered features, along with any domain knowledge or research insights driving feature selection.

## 3. **Temporal Metadata:**
- **Temporal Aggregations:** Maintain metadata on temporal aggregations used for generating time-based features, such as rolling statistics, lagged variables, and seasonal trends.
- **Time Granularity:** Specify the time granularity of the data, whether it is hourly, daily, or monthly, to ensure alignment with the model's prediction intervals.

## 4. **Model Training Metadata:**
- **Hyperparameter Tuning:** Document hyperparameters tuned during model training, including the selected values and reasons behind parameter choices.
- **Model Evaluation Metrics:** Capture the evaluation metrics used to assess model performance, such as Mean Absolute Error (MAE) or Root Mean Squared Error (RMSE).

## 5. **Data Versioning:**
- **Data Pipeline Versions:** Track versions of data preprocessing pipelines to ensure reproducibility and traceability of data transformations.
- **Feature Set Versioning:** Maintain versions of feature sets used for model training to facilitate model retraining and comparison.

## 6. **Integration with Model Serving:**
- **Input Data Schema:** Define the schema for input data expected by the deployed model to ensure data compatibility and consistency during inference.
- **Output Data Format:** Specify the format of model predictions to facilitate downstream integration with decision-making systems and reporting tools.

By implementing robust metadata management practices tailored to the unique demands of the Fishery Stock Assessment Model, Pesquera Exalmar can enhance data transparency, traceability, and interpretability, ultimately leading to more informed decision-making and sustainable fishing practices.

# Data Challenges and Preprocessing Strategies for Fishery Stock Assessment Model

## Specific Data Problems:
### 1. **Missing Data:**
- **Issue:** Incomplete or missing oceanographic data entries could hinder the model's ability to make accurate predictions.
- **Strategy:** Impute missing values using techniques like mean imputation or forward/backward fill based on the temporal nature of the data.

### 2. **Noise in Data:**
- **Issue:** Noisy measurements in oceanographic data may introduce inconsistencies and bias in the model.
- **Strategy:** Apply noise reduction techniques such as smoothing filters or outlier detection to clean the data before training the model.

### 3. **Seasonal Variations:**
- **Issue:** Uneven seasonal distribution of fish stock data could lead to biased model predictions.
- **Strategy:** Normalize seasonal variations by aggregating data over consistent time intervals or incorporating seasonal trend features in the model.

### 4. **Non-Stationarity:**
- **Issue:** Non-stationary patterns in oceanographic data may affect the model's ability to generalize across different time periods.
- **Strategy:** Employ techniques like differencing or detrending to make the data stationary and stabilize its statistical properties for modeling.

## Data Preprocessing Strategies:
### 1. **Quality Control Checks:**
- **Strategy:** Conduct rigorous data quality checks to identify and correct anomalies, outliers, and inconsistencies in the oceanographic data before preprocessing.

### 2. **Feature Selection:**
- **Strategy:** Use domain knowledge and feature importance analysis to select relevant features that have a significant impact on fish stock levels, eliminating irrelevant or redundant variables.

### 3. **Temporal Aggregations:**
- **Strategy:** Aggregate raw data into meaningful time intervals to capture temporal patterns effectively and improve the model's ability to learn dynamic relationships.

### 4. **Normalization and Scaling:**
- **Strategy:** Scale numerical features to a common range to prevent bias and gradient explosion during model training, ensuring all features contribute proportionally to the model predictions.

### 5. **Handling Temporal Dynamics:**
- **Strategy:** Incorporate lagged variables, rolling statistics, and other temporal features to capture temporal dependencies and long-term trends in the oceanographic data for better prediction accuracy.

### 6. **Data Imbalance Handling:**
- **Strategy:** Address data imbalance issues in fish stock data by using techniques like oversampling, undersampling, or synthetic data generation to ensure the model learns from all classes effectively.

By strategically employing these data preprocessing practices tailored to the unique challenges of the Fishery Stock Assessment Model, Pesquera Exalmar can ensure the data remains robust, reliable, and conducive to training high-performing machine learning models for sustainable fishing practices and optimal fish stock assessment.

```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

# Load the raw oceanographic data
oceanographic_data = pd.read_csv('oceanographic_data.csv')

# Check and handle missing data
missing_values_count = oceanographic_data.isnull().sum()
imputer = SimpleImputer(strategy='mean')
oceanographic_data_filled = pd.DataFrame(imputer.fit_transform(oceanographic_data), columns=oceanographic_data.columns)

# Feature Engineering: Normalize numerical features
scaler = MinMaxScaler()
numerical_features = oceanographic_data_filled.select_dtypes(include=['float64', 'int64']).columns
oceanographic_data_normalized = oceanographic_data_filled.copy()
oceanographic_data_normalized[numerical_features] = scaler.fit_transform(oceanographic_data_filled[numerical_features])

# Feature Selection: Choose relevant features based on domain knowledge
selected_features = ['sea_surface_temperature', 'chlorophyll_levels', 'salinity', 'oxygen_levels']

# Temporal Aggregations: Aggregate data over monthly intervals
oceanographic_data_monthly = oceanographic_data_normalized.resample('M').mean()

# Split data into training and testing sets
X = oceanographic_data_monthly[selected_features]
y = oceanographic_data_monthly['fish_stock_levels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save preprocessed data
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)
```

## Code Comments:
1. **Data Loading:** Read in the raw oceanographic data from a CSV file.
2. **Missing Data Imputation:** Utilize mean imputation to handle missing values in the data.
3. **Normalization:** Scale numerical features using Min-Max scaling to bring all features to a common range.
4. **Feature Selection:** Choose relevant features crucial for fish stock assessment based on domain knowledge and feature importance analysis.
5. **Temporal Aggregations:** Aggregate data over monthly intervals to capture seasonal trends and reduce temporal noise.
6. **Data Splitting:** Split the preprocessed data into training and testing sets for model evaluation.
7. **Save Preprocessed Data:** Save the preprocessed data sets (X_train, X_test, y_train, y_test) to CSV files for future model training and analysis.

By following this tailored data preprocessing code file, Pesquera Exalmar can ensure that the oceanographic data is prepared effectively for model training and analysis, aligning with the specific needs and characteristics of the Fishery Stock Assessment Model project.

# Modeling Strategy for Fishery Stock Assessment Model

For the Fishery Stock Assessment Model for Pesquera Exalmar, a Long Short-Term Memory (LSTM) neural network is recommended as the primary modeling approach due to its ability to capture long-term dependencies in sequential data and handle the temporal nature of oceanographic data effectively. LSTM networks are well-suited for time-series forecasting tasks, making them ideal for predicting fish stock levels based on historical oceanographic data.

## Recommended Modeling Strategy:
1. **LSTM Architecture Design:**
   - Design a deep LSTM network with multiple layers to capture complex patterns in the oceanographic data and learn from sequential information effectively.

2. **Sequence Generation:**
   - Create input sequences and target sequences from the preprocessed oceanographic data to train the LSTM model on historical patterns and predict future fish stock levels.

3. **Hyperparameter Tuning:**
   - Optimize hyperparameters such as the number of LSTM layers, number of units in each layer, dropout rates, and learning rate to enhance the model's performance and generalization capabilities.

4. **Temporal Attention Mechanism:**
   - Incorporate a temporal attention mechanism in the LSTM architecture to focus on relevant time steps and features, improving the model's ability to weight and interpret input data dynamically.

5. **Ensemble Learning:**
   - Implement ensemble learning techniques by training multiple LSTM models with different initializations or architectures and combining their predictions to boost overall model performance and reduce variance.

6. **Regularization Techniques:**
   - Apply regularization techniques like L2 regularization or dropout layers to prevent overfitting and improve the model's robustness in handling noisy oceanographic data.

7. **Model Interpretability Analysis:**
   - Conduct model interpretability analysis to understand the contributions of different features and time steps towards predicting fish stock levels, providing insights for decision-making and further model refinement.

8. **Crucial Step:**
   - **Temporal Feature Encoding:**
     - The most crucial step in this modeling strategy is the effective encoding of time-related features, such as lagged variables, seasonal trends, and rolling statistics. By accurately capturing temporal dependencies and patterns in the oceanographic data, the LSTM model can make precise predictions about fish stock levels over time, aligning with the project's overarching goal of ensuring sustainable fishing practices.

By emphasizing the encoding of temporal features within the LSTM architecture, Pesquera Exalmar can develop a robust and accurate Fishery Stock Assessment Model that leverages the unique characteristics of oceanographic data to predict fish stock levels effectively, ultimately facilitating sustainable fishing practices and informed decision-making.

# Tools and Technologies Recommendations for Data Modeling in Fishery Stock Assessment Model

To effectively implement the modeling strategy and address the specific data needs of the Fishery Stock Assessment Model for Pesquera Exalmar, the following tools and technologies are recommended:

## 1. **TensorFlow with Keras**
- **Description:** TensorFlow with Keras offers a flexible deep learning framework for building and training LSTM models to predict fish stock levels accurately based on oceanographic data.
- **Integration:** Integrates seamlessly with Python programming language and supports GPU acceleration for faster model training.
- **Key Features:**
   - High-level APIs for building LSTM architectures.
   - TensorBoard for model visualization and performance monitoring.
   - Customizable callbacks for early stopping and model checkpointing.
- **Documentation:** [TensorFlow](https://www.tensorflow.org/) | [Keras](https://keras.io/)

## 2. **scikit-learn**
- **Description:** scikit-learn provides a wide range of machine learning algorithms and tools for preprocessing, model evaluation, and ensemble learning techniques to augment LSTM models.
- **Integration:** Easily integrates with NumPy and Pandas for data manipulation and preprocessing.
- **Key Features:**
   - Model selection and hyperparameter tuning capabilities.
   - Data preprocessing functions like scaling, imputation, and encoding.
   - Ensemble methods for combining LSTM model predictions.
- **Documentation:** [scikit-learn](https://scikit-learn.org/stable/)

## 3. **AWS SageMaker**
- **Description:** AWS SageMaker is a cloud-based platform that offers scalable infrastructure for training and deploying machine learning models, suitable for handling large-scale oceanographic data processing and analysis.
- **Integration:** Seamlessly integrates with AWS S3 for data storage and AWS Lambda for serving model predictions through APIs.
- **Key Features:**
   - Built-in algorithms for model training and hyperparameter optimization.
   - Model hosting for deploying LSTM models as web services.
   - Automatic model scaling and resource management.
- **Documentation:** [AWS SageMaker](https://aws.amazon.com/sagemaker/)

## 4. **TensorBoard**
- **Description:** TensorBoard is a visualization toolkit for TensorFlow that enables tracking and visualizing model training metrics, model graphs, and embeddings, providing insights into LSTM model performance and interpretation.
- **Integration:** Compatible with TensorFlow and Keras models, allowing seamless visualization of LSTM model training progress.
- **Key Features:**
   - Visualization of training and validation metrics for optimizing hyperparameters.
   - Model graph visualization for LSTM architecture understanding.
   - Embedding Projector for exploring high-dimensional data representations.
- **Documentation:** [TensorBoard](https://www.tensorflow.org/tensorboard)

By leveraging these tools and technologies tailored to the data modeling needs of the Fishery Stock Assessment Model, Pesquera Exalmar can enhance efficiency, accuracy, and scalability in predicting fish stock levels and guiding sustainable fishing practices. Each tool plays a crucial role in different stages of the modeling process, ensuring a comprehensive and effective solution to the fisheries manager's pain point.

To generate a large fictitious dataset that mimics real-world oceanographic data for the Fishery Stock Assessment Model, we can use Python with NumPy and Pandas to create synthetic data with variability across key features. The script will incorporate seasonal trends, lagged variables, and noisy fluctuations to simulate realistic conditions for model training and validation.

```python
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Generate synthetic time-series data for oceanographic features
np.random.seed(42)

# Define time range
start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 12, 31)
date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days)]

# Generate synthetic features: sea surface temperature, chlorophyll levels, salinity, oxygen levels, fish stock levels
sea_surface_temperature = np.sin(2 * np.pi * np.arange(len(date_range)) / 365) * 10 + np.random.normal(0, 1, len(date_range))
chlorophyll_levels = np.cos(2 * np.pi * np.arange(len(date_range)) / 365) * 5 + np.random.normal(0, 0.5, len(date_range))
salinity = np.random.uniform(30, 35, len(date_range))
oxygen_levels = np.random.normal(5, 0.5, len(date_range))
fish_stock_levels = sea_surface_temperature + chlorophyll_levels + salinity + oxygen_levels + np.random.normal(0, 1, len(date_range))

# Create a synthetic dataset
synthetic_data = pd.DataFrame({
    'Date': date_range,
    'Sea_Surface_Temperature': sea_surface_temperature,
    'Chlorophyll_Levels': chlorophyll_levels,
    'Salinity': salinity,
    'Oxygen_Levels': oxygen_levels,
    'Fish_Stock_Levels': fish_stock_levels
})

# Add noise to simulate real-world variability
synthetic_data['Sea_Surface_Temperature'] += np.random.normal(0, 0.5, len(date_range))
synthetic_data['Chlorophyll_Levels'] += np.random.normal(0, 0.2, len(date_range))
synthetic_data['Salinity'] += np.random.normal(0, 0.1, len(date_range))
synthetic_data['Oxygen_Levels'] += np.random.normal(0, 0.1, len(date_range))

# Shuffle the dataset
synthetic_data = synthetic_data.sample(frac=1).reset_index(drop=True)

# Save synthetic dataset to CSV
synthetic_data.to_csv('synthetic_oceanographic_data.csv', index=False)
```

This script generates a synthetic dataset with fictitious oceanographic features, incorporating noise to reflect real-world variability. The dataset includes sea surface temperature, chlorophyll levels, salinity, oxygen levels, and fish stock levels over a temporal range, simulating diverse conditions for model training and validation. The dataset aligns with the project's modeling strategy and can be seamlessly integrated into the model training pipeline for accurate prediction of fish stock levels.

Below is a sample excerpt of the mocked oceanographic dataset tailored to the Fishery Stock Assessment Model for Pesquera Exalmar, showcasing a few rows of synthetic data with feature names and types:

```plaintext
| Date       | Sea_Surface_Temperature | Chlorophyll_Levels | Salinity | Oxygen_Levels | Fish_Stock_Levels |
|------------|-------------------------|--------------------|----------|--------------|-------------------|
| 2010-01-01 | 12.345                  | 3.456              | 31.2     | 5.8          | 53.42             |
| 2010-01-02 | 12.123                  | 3.678              | 31.1     | 5.9          | 53.87             |
| 2010-01-03 | 11.987                  | 3.789              | 31.3     | 6.0          | 53.25             |
| 2010-01-04 | 11.895                  | 3.543              | 31.1     | 5.8          | 52.91             |
| 2010-01-05 | 12.432                  | 3.467              | 31.0     | 5.9          | 53.65             |
```

In this sample, each row represents a daily observation of sea surface temperature, chlorophyll levels, salinity, oxygen levels, and corresponding fish stock levels. The data is structured in a tabular format with clear delineation of feature names and their respective numerical types. This representation aligns with the standard CSV format for model ingestion, ensuring seamless integration of the mocked data into the training pipeline for the Fishery Stock Assessment Model.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load preprocessed dataset
data = pd.read_csv('preprocessed_data.csv')

# Split data into features and target
X = data.drop(columns=['Fish_Stock_Levels'])
y = data['Fish_Stock_Levels']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape data for LSTM input [samples, time steps, features]
X_train = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Define LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Evaluate the model
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
train_rmse = mean_squared_error(y_train, train_pred, squared=False)
test_rmse = mean_squared_error(y_test, test_pred, squared=False)
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# Save the trained model
model.save('fishery_stock_assessment_model.h5')
```

## Code Comments:
1. **Data Loading:** Load the preprocessed dataset containing features and target variable (fish stock levels).
2. **Data Splitting:** Split the data into training and testing sets for model training and evaluation.
3. **Data Reshaping:** Reshape the data to fit the input shape required by the LSTM model.
4. **Model Definition:** Define the LSTM model architecture with input shape, LSTM layer, and output layer.
5. **Model Training:** Train the LSTM model on the training data with specified epochs and batch size.
6. **Model Evaluation:** Evaluate the model performance on both training and testing sets using Root Mean Squared Error.
7. **Model Saving:** Save the trained LSTM model for future production deployment.

This code snippet adheres to best practices for readability and maintainability in production environments, following standard conventions for model training and evaluation processes commonly adopted in large tech companies.

# Deployment Plan for Fishery Stock Assessment Model

To deploy the Machine Learning model for the Fishery Stock Assessment project, tailored to the unique demands of Pesquera Exalmar, we recommend the following step-by-step deployment plan:

## 1. **Pre-Deployment Checks:**
- **Ensure Model Readiness:** Verify that the LSTM model is trained, evaluated, and saved in a production-ready format.
- **Data Integrity Check:** Validate that the preprocessed data used for training the model is consistent and up-to-date.

## 2. **Model Serialization and Exporting:**
- **Tool:** TensorFlow/Keras for model serialization.
- **Steps:** Serialize the trained LSTM model to a file format suitable for deployment, such as a SavedModel (.pb) or Hierarchical Data Format (.h5).
- **Documentation:** [TensorFlow Model Saving](https://www.tensorflow.org/guide/keras/save_and_serialize), [Keras Model Saving](https://keras.io/guides/serialization/)

## 3. **Setup Cloud Infrastructure:**
- **Tool:** Amazon Web Services (AWS) for cloud hosting.
- **Steps:** Set up an AWS account, create an S3 bucket for storing model artifacts, and configure necessary permissions for model access.
- **Documentation:** [AWS Documentation](https://docs.aws.amazon.com/)

## 4. **Model Deployment:**
- **Tool:** AWS SageMaker for model deployment.
- **Steps:** Deploy the serialized LSTM model on AWS SageMaker, create an endpoint for serving predictions, and ensure endpoint availability and scalability.
- **Documentation:** [AWS SageMaker Deploy Model](https://docs.aws.amazon.com/sagemaker/)

## 5. **API Integration:**
- **Tool:** AWS API Gateway for creating APIs.
- **Steps:** Create an API using AWS API Gateway to interface with the model endpoint, enabling easy access for data input and receiving predictions.
- **Documentation:** [AWS API Gateway Documentation](https://docs.aws.amazon.com/apigateway/)

## 6. **Testing and Monitoring:**
- **Tool:** AWS CloudWatch for monitoring.
- **Steps:** Set up monitoring and logging using AWS CloudWatch to track model performance, behavior, and any anomalies in real-time deployment.
- **Documentation:** [AWS CloudWatch Documentation](https://docs.aws.amazon.com/cloudwatch/)

## 7. **Live Environment Integration:**
- **Tool:** AWS Lambda for serverless computing.
- **Steps:** Integrate the model API with existing systems or applications using AWS Lambda functions for on-demand computing capabilities and resource management.
- **Documentation:** [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)

By following this deployment plan with the recommended tools and platforms, Pesquera Exalmar can successfully transition the Fishery Stock Assessment Model from development to production, ensuring seamless integration, scalability, and real-time monitoring of the machine learning model to support sustainable fishing practices efficiently.

```Dockerfile
# Use the official Tensorflow with GPU base image
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory
WORKDIR /app

# Copy the model file and dependencies
COPY fishery_stock_assessment_model.h5 /app
COPY requirements.txt /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port
EXPOSE 8080

# Command to start the model serving
CMD ["python", "model_serving.py"]
```

In this optimized Dockerfile for the Fishery Stock Assessment Model deployment, specific configurations are tailored to ensure performance and scalability, such as utilizing the official TensorFlow with GPU base image for enhanced computation, setting the working directory, installing dependencies from a requirements file, exposing a specific port for external access, and defining the command to start the model serving script (`model_serving.py`).

## User Groups and User Stories

### 1. **Fisheries Manager:**
- **User Story:** As a Fisheries Manager, I struggle to sustain fish populations while maximizing catch due to inaccurate fish stock assessments.
- **Solution:** The application provides accurate predictions of fish stock levels based on oceanographic data, enabling informed decision-making on fishing practices to ensure sustainability.
- **Facilitating Component:** The LSTM model trained on preprocessed oceanographic data facilitates accurate fish stock predictions to guide sustainable fishing practices.

### 2. **Fishermen:**
- **User Story:** As a Fisherman, I face uncertainties in identifying abundant fishing grounds, impacting catch efficiency and profitability.
- **Solution:** The application predicts fish stock levels in different fishing areas, helping fishermen locate abundant fish populations and optimize catch for increased profitability.
- **Facilitating Component:** The model serving script in the Docker container provides real-time predictions for fishing ground selection.

### 3. **Environmental Researchers:**
- **User Story:** As an Environmental Researcher, I struggle to analyze complex oceanographic data for fish population studies, hindering research progress.
- **Solution:** The application processes and analyzes oceanographic data using LSTM models, supporting research on fish populations and environmental impact studies.
- **Facilitating Component:** The data preprocessing script and LSTM model enhance the analysis and interpretation of oceanographic data for research purposes.

### 4. **Regulatory Authorities:**
- **User Story:** As a Regulatory Authority, I find it challenging to enforce sustainable fishing regulations without accurate fish stock assessments.
- **Solution:** The application offers reliable fish stock predictions to support regulatory decisions, ensuring compliance with sustainable fishing practices.
- **Facilitating Component:** The deployment pipeline and monitoring setup enable real-time access to accurate fish stock assessments for regulatory decision-making.

By addressing the diverse pain points of user groups through specific user stories and showcasing how the Fishery Stock Assessment Model benefits each group, the project demonstrates its value proposition in promoting sustainable fishing practices and informed decision-making for all stakeholders involved.