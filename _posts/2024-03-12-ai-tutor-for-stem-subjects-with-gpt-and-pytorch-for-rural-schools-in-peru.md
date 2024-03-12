---
title: AI Tutor for STEM Subjects with GPT and PyTorch for Rural Schools in Peru - Project Objective: Offer accessible STEM education resources to rural areas, solution is to deploy ChatGPT-powered AI tutors that provide explanations, solve problems, and interact in real-time, bridging educational gaps.
date: 2024-03-12
permalink: posts/ai-tutor-for-stem-subjects-with-gpt-and-pytorch-for-rural-schools-in-peru
layout: article
---

### Project Focus and Target Variable

#### Primary Focus:
The primary focus of the project is to deploy AI tutors powered by ChatGPT and PyTorch to offer accessible STEM education resources to rural schools in Peru. The AI tutors are designed to provide explanations, solve problems, and interact in real-time, aiming to bridge educational gaps in these areas.

#### Target Variable Name:
- **Engagement Score**
  
#### Importance of the Target Variable:
The target variable, "Engagement Score," serves as a key metric to measure the effectiveness and impact of the AI tutors in engaging students in STEM subjects. This score encapsulates how actively and attentively students interact with the AI tutor during learning sessions. By monitoring the Engagement Score, educators and developers can assess the level of student involvement, interest, and receptiveness to the AI tutor's teaching methods.

#### Example Values and Decision-making:
- **Example Value 1:**
  - Engagement Score: 75%
  - Interpretation: A high Engagement Score of 75% indicates that students are actively participating, asking questions, and staying focused during the AI tutor sessions. This suggests that the AI tutor is effectively capturing students' attention and maintaining their interest in STEM topics.
  - Decision-making: Based on this value, educators can conclude that the AI tutor is successfully engaging students and tailor future sessions to maintain or improve this level of engagement.

- **Example Value 2:**
  - Engagement Score: 40%
  - Interpretation: A lower Engagement Score of 40% implies that students may be less interested, less interactive, or facing difficulties following the AI tutor's explanations and solutions. It indicates a need to review and adjust the teaching approach to enhance student engagement and comprehension.
  - Decision-making: Upon observing this value, educators can analyze the areas where students show less engagement, refine the content delivery, introduce interactive activities, or provide additional support to boost student involvement and understanding.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate mock data
np.random.seed(42)
data = {
    'Time_Spent': np.random.uniform(low=0.5, high=3.0, size=1000),  # Amount of time spent by the student on the AI tutor (hours)
    'Questions_Asked': np.random.randint(0, 10, size=1000),  # Number of questions asked by the student during the session
    'Engagement_Score': np.random.randint(50, 100, size=1000)  # Engagement Score (target variable)
}
df = pd.DataFrame(data)

# Define features and target variable
X = df[['Time_Spent', 'Questions_Asked']]
y = df['Engagement_Score']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict the target variable for the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance using Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print("Predicted Engagement Scores for the test set:")
print(y_pred)
```

In this Python script:
- **Mock Data Generation**: We generate mock data representing students' time spent on the AI tutor, questions asked, and the engagement score.
- **Model Training**: We use a Random Forest Regressor to train the model on the generated data.
- **Predictions**: We predict the engagement scores for the test set using the trained model.
- **Model Evaluation**: We evaluate the model's performance using the Mean Squared Error metric.

This script provides a practical example of how to create, train, and evaluate a machine learning model for predicting the engagement score of students interacting with the AI tutor based on their time spent and questions asked.

### Secondary Target Variable: Knowledge Retention Score

#### Importance and Role:
The secondary target variable, "Knowledge Retention Score," could play a critical role in enhancing our predictive model's accuracy and insights. This score represents how well students retain and apply the STEM knowledge gained through interactions with the AI tutor. It reflects the long-term effectiveness of the educational intervention provided by the AI tutor in improving students' understanding and retention of the subject matter.

#### Complement to Primary Target Variable:
The Knowledge Retention Score complements the Engagement Score by providing a holistic view of the AI tutor's impact on students' learning outcomes. While the Engagement Score measures immediate engagement and interaction levels, the Knowledge Retention Score evaluates the lasting benefit and practical application of the knowledge acquired. By considering both scores, educators and developers can gain deeper insights into the overall effectiveness of the AI tutor in enhancing students' learning experiences and academic performance in STEM subjects.

#### Example Values and Decision-making:
- **Example Value 1:**
  - Knowledge Retention Score: 85%
  - Interpretation: A high Knowledge Retention Score of 85% indicates that students demonstrate strong retention and application of the STEM concepts learned through the AI tutor sessions. This suggests that the AI tutor's teaching methods are effective in facilitating long-term understanding and knowledge retention.
  - Decision-making: Educators can leverage this value to identify successful teaching strategies employed by the AI tutor and reinforce similar approaches to optimize knowledge retention across different topics and student groups.

- **Example Value 2:**
  - Knowledge Retention Score: 60%
  - Interpretation: A lower Knowledge Retention Score of 60% implies that students may struggle with retaining and applying the STEM knowledge acquired during the AI tutor sessions. It signals a need to revisit the content delivery, incorporate more interactive exercises, or provide additional resources to improve knowledge retention.
  - Decision-making: Upon observing this value, educators can reflect on the areas where students exhibit lower retention levels, adjust the teaching approach accordingly, and customize learning materials to enhance knowledge retention and reinforce understanding in challenging concepts.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate mock data
np.random.seed(42)
data = {
    'Time_Spent': np.random.uniform(low=0.5, high=3.0, size=1000),
    'Questions_Asked': np.random.randint(0, 10, size=1000),
    'Engagement_Score': np.random.randint(50, 100, size=1000),
    'Knowledge_Retention_Score': np.random.randint(50, 100, size=1000)
}
df = pd.DataFrame(data)

# Define features and target variables
X = df[['Time_Spent', 'Questions_Asked']]
y_primary = df['Engagement_Score']
y_secondary = df['Knowledge_Retention_Score']

# Split data into training and test sets
X_train, X_test, y_train_primary, y_test_primary, y_train_secondary, y_test_secondary = train_test_split(X, y_primary, y_secondary, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor models for both target variables
model_primary = RandomForestRegressor(random_state=42)
model_secondary = RandomForestRegressor(random_state=42)
model_primary.fit(X_train, y_train_primary)
model_secondary.fit(X_train, y_train_secondary)

# Predict the target variables for the test set
y_pred_primary = model_primary.predict(X_test)
y_pred_secondary = model_secondary.predict(X_test)

# Evaluate the models' performance using Mean Squared Error
mse_primary = mean_squared_error(y_test_primary, y_pred_primary)
mse_secondary = mean_squared_error(y_test_secondary, y_pred_secondary)

print(f"Mean Squared Error for Primary Target Variable: {mse_primary}")
print("Predicted Engagement Scores for the test set:")
print(y_pred_primary)

print(f"Mean Squared Error for Secondary Target Variable: {mse_secondary}")
print("Predicted Knowledge Retention Scores for the test set:")
print(y_pred_secondary)
```

In this Python script:
- **Mock Data Generation**: We generate mock data representing students' time spent, questions asked, engagement score, and knowledge retention score.
- **Model Training**: We create and train separate Random Forest Regressor models for both the primary target variable (Engagement Score) and the secondary target variable (Knowledge Retention Score).
- **Predictions**: We predict the target variables for the test set using the trained models.
- **Model Evaluation**: We evaluate the models' performance using the Mean Squared Error metric for both target variables.

This script showcases how to develop and evaluate machine learning models for predicting both the Engagement Score and Knowledge Retention Score based on student interactions with the AI tutor.

### Third Target Variable: Problem-Solving Proficiency Score

#### Importance and Role:
The third target variable, "Problem-Solving Proficiency Score," plays a crucial role in assessing students' ability to apply STEM concepts and critical thinking skills to solve complex problems. This score evaluates how effectively students can analyze, strategize, and implement solutions to real-world challenges in STEM disciplines. By measuring problem-solving proficiency, educators can gain insights into students' practical application of acquired knowledge and skills.

#### Complement to Primary and Secondary Target Variables:
The Problem-Solving Proficiency Score complements the Engagement Score and Knowledge Retention Score by focusing on students' application and practical reasoning abilities. While the Engagement Score reflects student engagement during learning sessions and the Knowledge Retention Score evaluates long-term understanding, the Problem-Solving Proficiency Score assesses students' problem-solving capabilities in real-world scenarios. By considering all three variables together, educators and developers can gain a comprehensive understanding of students' overall learning outcomes and adaptive skills in STEM subjects.

#### Example Values and Decision-making:
- **Example Value 1:**
  - Problem-Solving Proficiency Score: 90%
  - Interpretation: A high Problem-Solving Proficiency Score of 90% indicates that students demonstrate strong problem-solving abilities and excel in applying STEM concepts to solve challenging problems. This suggests that the AI tutor effectively enhances students' critical thinking skills and prepares them for real-world applications in STEM fields.
  - Decision-making: Educators can use this value to identify students with advanced problem-solving skills, tailor additional challenges or projects to further develop their abilities, and recognize the effectiveness of the AI tutor in fostering advanced problem-solving proficiency.

- **Example Value 2:**
  - Problem-Solving Proficiency Score: 60%
  - Interpretation: A lower Problem-Solving Proficiency Score of 60% implies that students may struggle with applying theoretical knowledge to practical problem-solving tasks. It signals a need to provide more hands-on activities, real-life examples, or interactive simulations to enhance students' problem-solving skills.
  - Decision-making: By analyzing this value, educators can identify areas where students face challenges in problem-solving, adjust the curriculum to include more problem-solving exercises, and offer personalized support to help students improve their problem-solving proficiency over time.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Generate mock data
np.random.seed(42)
data = {
    'Time_Spent': np.random.uniform(low=0.5, high=3.0, size=1000),
    'Questions_Asked': np.random.randint(0, 10, size=1000),
    'Engagement_Score': np.random.randint(50, 100, size=1000),
    'Knowledge_Retention_Score': np.random.randint(50, 100, size=1000),
    'Problem_Solving_Proficiency_Score': np.random.randint(50, 100, size=1000)
}
df = pd.DataFrame(data)

# Define features and target variables
X = df[['Time_Spent', 'Questions_Asked']]
y_primary = df['Engagement_Score']
y_secondary = df['Knowledge_Retention_Score']
y_third = df['Problem_Solving_Proficiency_Score']

# Split data into training and test sets
X_train, X_test, y_train_primary, y_test_primary, y_train_secondary, y_test_secondary, y_train_third, y_test_third = train_test_split(X, y_primary, y_secondary, y_third, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor models for all target variables
model_primary = RandomForestRegressor(random_state=42)
model_secondary = RandomForestRegressor(random_state=42)
model_third = RandomForestRegressor(random_state=42)
model_primary.fit(X_train, y_train_primary)
model_secondary.fit(X_train, y_train_secondary)
model_third.fit(X_train, y_train_third)

# Predict the target variables for the test set
y_pred_primary = model_primary.predict(X_test)
y_pred_secondary = model_secondary.predict(X_test)
y_pred_third = model_third.predict(X_test)

# Evaluate the models' performance using Mean Squared Error
mse_primary = mean_squared_error(y_test_primary, y_pred_primary)
mse_secondary = mean_squared_error(y_test_secondary, y_pred_secondary)
mse_third = mean_squared_error(y_test_third, y_pred_third)

print(f"Mean Squared Error for Primary Target Variable: {mse_primary}")
print("Predicted Engagement Scores for the test set:")
print(y_pred_primary)

print(f"Mean Squared Error for Secondary Target Variable: {mse_secondary}")
print("Predicted Knowledge Retention Scores for the test set:")
print(y_pred_secondary)

print(f"Mean Squared Error for Third Target Variable: {mse_third}")
print("Predicted Problem-Solving Proficiency Scores for the test set:")
print(y_pred_third)
```

In this Python script:
- **Mock Data Generation**: We generate mock data representing students' time spent, questions asked, engagement score, knowledge retention score, and problem-solving proficiency score.
- **Model Training**: We create and train separate Random Forest Regressor models for all three target variables: Engagement Score, Knowledge Retention Score, and Problem-Solving Proficiency Score.
- **Predictions**: We predict the target variables for the test set using the trained models.
- **Model Evaluation**: We evaluate the models' performance using the Mean Squared Error metric for each target variable.

This script demonstrates how to develop and evaluate machine learning models for predicting multiple target variables related to student engagement, knowledge retention, and problem-solving proficiency in STEM subjects using data from interactions with the AI tutor.

### User Groups and User Stories

#### 1. **Students in Rural Schools**
   - **User Story**: Maria, a high school student in a rural school in Peru, often struggles with understanding complex STEM concepts in Physics class. She finds it challenging to retain the information taught in class and apply it to solve problems during exams.
   - **Pain Point**: Maria's pain point lies in her difficulty in retaining and applying the learned STEM knowledge effectively.
   - **Benefit**: By improving her Knowledge Retention Score through interactions with the AI tutor, Maria can enhance her ability to remember and utilize the learned concepts, leading to better academic performance and increased confidence in tackling Physics problems.
   - **Project Component**: The provision of real-time explanations and problem-solving assistance by the ChatGPT-powered AI tutor facilitates Maria's understanding of challenging topics, aiding in improving her Knowledge Retention Score.

#### 2. **Teachers in Rural Schools**
   - **User Story**: Juan, a STEM teacher in a rural school, faces the challenge of engaging and motivating students who have varying levels of interest and understanding in Mathematics.
   - **Pain Point**: Juan struggles to keep all students engaged and actively participating in classroom activities, leading to gaps in students' learning outcomes.
   - **Benefit**: By tracking the Engagement Scores of students using the AI tutor, Juan can identify students who require additional support or personalized attention, enabling him to tailor teaching strategies and interventions to boost student engagement and participation.
   - **Project Component**: The AI tutor's real-time interactions and engagement monitoring feature support Juan in addressing student engagement issues and adjusting teaching methods to enhance student participation and learning outcomes.

#### 3. **Educational Administrators in Rural Areas**
   - **User Story**: Ana, an educational administrator in a rural region of Peru, is concerned about the overall academic performance of students in STEM subjects across different schools. She seeks data-driven insights to evaluate the effectiveness of educational interventions.
   - **Pain Point**: Ana faces challenges in assessing the impact of teaching initiatives and identifying areas for improvement in STEM education within rural schools.
   - **Benefit**: By analyzing the Problem-Solving Proficiency Scores generated by the AI tutor, Ana gains valuable insights into students' abilities to apply STEM concepts in practical scenarios, enabling her to assess the effectiveness of the AI tutor in enhancing problem-solving skills and fostering critical thinking.
   - **Project Component**: The AI tutor's feature that evaluates and measures students' problem-solving proficiency provides Ana with data-driven indicators to assess the success of the educational program and make informed decisions on future interventions and resource allocations.

By addressing the pain points of diverse user groups through the insights offered by different target variables, the AI Tutor for STEM Subjects project effectively caters to the specific needs and challenges faced by students, teachers, and educational administrators in rural schools in Peru, ultimately contributing to bridging educational gaps and improving STEM education accessibility and outcomes.

### User-Centric Narrative: 

#### [User_Name]: Sofia, a High School Student in a Rural School

#### User Challenge:
Sofia, a high school student in a rural school in Peru, struggles with understanding and applying complex STEM concepts in Mathematics class. Despite her dedication to learning, she often faces difficulties in grasping abstract mathematical principles.

#### Pain Point:
Sofia's main pain point is her challenge in retaining the learned mathematical concepts, leading to a lack of confidence in solving advanced problems effectively.

#### Negative Impact:
This difficulty in understanding and applying mathematical concepts negatively impacts Sofia's grades and academic performance, causing frustration and hindering her passion for STEM subjects.

#### Project Solution: AI Tutor for STEM Subjects with GPT and PyTorch

#### Target Variable: Problem-Solving Proficiency Score

- **Solution Feature**: The AI tutor leverages machine learning to assess Sofia's problem-solving proficiency in Mathematics, providing tailored feedback and guidance in tackling complex mathematical problems.
- **Solution Benefit**: By analyzing Sofia's Problem-Solving Proficiency Score, the AI tutor offers personalized problem-solving strategies, enhancing her critical thinking skills and boosting her confidence in solving challenging math problems.

#### Sofia's Journey:
Sofia decides to engage with the AI tutor and receives a Problem-Solving Proficiency Score of 85%. The system recommends practicing more problem-solving exercises regularly to strengthen her skills and suggests breaking down complex problems into manageable steps.

Initially hesitant and curious, Sofia decides to follow the recommendation and dedicates time to practicing problem-solving techniques as advised by the AI tutor. As a result, she experiences a breakthrough in understanding difficult concepts, improving her problem-solving abilities, and achieving better grades in Mathematics.

#### Positive Impacts:
- Increased confidence in tackling complex mathematical problems
- Improved academic performance in Mathematics
- Enhanced problem-solving skills applicable beyond the classroom

#### Reflection:
The insights derived from the AI tutor's Problem-Solving Proficiency Score empowered Sofia to adopt a structured approach to problem-solving, leading to a significant improvement in her mathematical abilities and overall academic success. This transformative experience showcases how machine learning applications can provide actionable guidance and personalized support, ultimately empowering individuals like Sofia to overcome educational challenges and excel in STEM subjects.

#### Broader Implications:
This project highlights the transformative power of data-driven decision-making in education. By utilizing machine learning to offer personalized guidance and support, individuals facing similar challenges to Sofia can benefit from targeted interventions, tailored learning strategies, and improved educational outcomes in STEM subjects. The innovative use of technology in education demonstrates the potential of AI-powered solutions to bridge educational gaps and empower learners in diverse settings.