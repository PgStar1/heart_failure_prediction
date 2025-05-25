# heart_failure_prediction
# Heart Failure Prediction Dashboard

This Python application uses machine learning and a web framework (Dash) to predict the likelihood of heart failure based on various health features. The model is built using K-Nearest Neighbors (KNN) and is designed to predict whether a patient is at risk of death due to heart failure.

## Key Features:
1. **User Input**: The dashboard collects input from the user for various health-related features such as age, anaemia, diabetes, blood pressure, and other medical factors that affect heart failure risk.
2. **Machine Learning Model**: The model uses K-Nearest Neighbors (KNN) classification to make predictions based on the provided input features. The model is trained using historical heart failure clinical records.
3. **Confusion Matrix Visualization**: After making a prediction, the dashboard displays a confusion matrix to illustrate the performance of the model. This matrix compares the predicted results to the true outcomes from the test data.
4. **Prediction Output**: The result shows whether the patient is predicted to survive or die from heart failure based on the input features.

## Data and Model Details:
- The model was trained using clinical records that include several key medical features such as `age`, `anaemia`, `creatinine_phosphokinase`, `diabetes`, `ejection_fraction`, `high_blood_pressure`, `platelets`, `serum_creatinine`, `serum_sodium`, `sex`, `smoking`, and `time`.
- The model uses K-Nearest Neighbors with the Manhattan distance metric, and the optimal number of neighbors (`k`) is selected based on cross-validation.
- The model is stored using the `joblib` library and is loaded into the application when running.

## How the Dashboard Works:
1. **User Inputs**: 
   - The user enters values for various health metrics, such as `age`, `anaemia`, `platelets`, etc.
   - The input is taken through Dash input fields, allowing numeric input for each health factor.
   
2. **Data Processing**:
   - After clicking the "Submit" button, the inputs are standardized using `StandardScaler` to ensure they are in the same range for model prediction.
   
3. **Prediction**:
   - The model is used to predict whether the patient will survive or die from heart failure based on the input features.
   
4. **Confusion Matrix**:
   - The confusion matrix is generated based on the test data, and a visual representation of the confusion matrix is displayed.
   
5. **Prediction Output**:
   - The app displays a message that shows whether the patient is predicted to "Survive" or "Die" based on the model’s prediction.

## Code Components:
1. **Machine Learning Model**:
   - The model is built using the `KNeighborsClassifier` from the `sklearn` library, and it is trained on heart failure clinical records.
   - The model is stored as `knn_model.pkl` using `joblib`.

2. **Dash Web Framework**:
   - Dash components are used to create the layout and interactivity of the dashboard.
   - The `dcc.Input` components are used for user input, and the `html.Div` component is used to structure the output and predictions.

3. **Confusion Matrix Plot**:
   - The confusion matrix is generated using `seaborn` and `matplotlib`, and the image is embedded in the webpage using base64 encoding.

## Technologies Used:
- **Python**: The core programming language used to implement the machine learning model and the web dashboard.
- **Dash**: A Python web framework used to create interactive web applications.
- **scikit-learn (sklearn)**: Used for machine learning algorithms (KNN classifier) and preprocessing.
- **Matplotlib and Seaborn**: Used to visualize the confusion matrix.
- **Joblib**: Used to store and load the trained machine learning model.

## Example Workflow:
1. The user enters their medical data into the input fields.
2. They click the "Submit" button to get a prediction.
3. The model makes a prediction and outputs whether the person is predicted to survive or not.
4. The confusion matrix is displayed as a heatmap to show how well the model performs.

## Running the Application:
- To run the application, ensure that the necessary libraries are installed:
  ```bash
  pip install dash dash-bootstrap-components scikit-learn pandas matplotlib seaborn joblib
