import sklearn
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
from sklearn import linear_model,preprocessing
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import joblib
####
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

raw_data_path = Path(__file__).parent.parent.joinpath('data','heart_failure_clinical_records.csv')
raw_data = pd.read_csv(raw_data_path)
pd.set_option('display.max_rows', raw_data.shape[0] + 1)
pd.set_option('display.max_columns', raw_data.shape[1] + 1)
#print(raw_data.shape)
#print(raw_data['sex'] )

"""
predict = 'DEATH_EVENT'

X = list(zip(raw_data['age'],raw_data['anaemia'],raw_data['creatinine_phosphokinase'],raw_data['diabetes'],raw_data['ejection_fraction'],raw_data['high_blood_pressure'],
    raw_data['platelets'],raw_data['serum_creatinine'],raw_data['serum_sodium'],raw_data['sex'],raw_data['smoking'],raw_data['time']))
y = raw_data['DEATH_EVENT']

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(X, y, test_size= 0.1, random_state=42)
print(len(y_train), len(y_test))  
model = KNeighborsClassifier(n_neighbors= 3, metric='manhattan')
model.fit(x_train, y_train)   
acc = model.score(x_test , y_test)
print(acc)

predicted  = model.predict(x_test)
cm  = confusion_matrix(y_test, predicted)
print(sum(cm))
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot= True)
plt.xlabel("pred")
plt.ylabel("trues")
#plt.show()

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Load your data
raw_data_path = Path(__file__).parent.parent.joinpath('data','heart_failure_clinical_records.csv')
raw_data = pd.read_csv(raw_data_path)  # Update with your actual data loading method

predict = 'DEATH_EVENT'
features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
            'sex', 'smoking', 'time']

X = raw_data[features].values
y = raw_data[predict].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define a range of k values to test
k_range = range(1, 31)
metrics = ['euclidean', 'manhattan', 'chebyshev']
metric_scores = {}

# Perform cross-validation for each metric and k
for metric in metrics:
    k_scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
        scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
        k_scores.append(scores.mean())
    metric_scores[metric] = k_scores

# Find the best metric and k
best_metric = max(metric_scores, key=lambda metric: max(metric_scores[metric]))
best_k = k_range[metric_scores[best_metric].index(max(metric_scores[best_metric]))]

print(f'The optimal metric is {best_metric} with k={best_k}')

# Train the model with the optimal metric and k
model = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
model.fit(x_train, y_train)

# Evaluate the model
acc = model.score(x_test, y_test)
print(f'Accuracy with {best_metric} metric and k={best_k}: {acc:.2f}')

# Make predictions
predicted = model.predict(x_test)

# Print a sample prediction
if len(predicted) > 5:
    print(f'Predicted: {predicted[5]}, Data: {x_test[5]}, Actual: {y_test[5]}')

# Print all predictions
for i in range(len(predicted)):
    print(f'Predicted: {predicted[i]}, Data: {x_test[i]}, Actual: {y_test[i]}')
"""

predict = 'DEATH_EVENT'
features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium',
            'sex', 'smoking', 'time']

X = raw_data[features].values
y = raw_data[predict].values

# Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Define a range of k values to test
k_range = range(1, 31)

# Store the scores for each k
k_scores = []

# Perform cross-validation for the Manhattan metric
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k, metric='manhattan')
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())

# Find the k with the highest average cross-validated accuracy
optimal_k = k_range[k_scores.index(max(k_scores))]
print(f'The optimal number of neighbors is {optimal_k} with the Manhattan distance metric')

# Train the model with the optimal k
model = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
model.fit(x_train, y_train)

# Evaluate the model
acc = model.score(x_test, y_test)
#print(f'Accuracy with Manhattan distance metric and k={optimal_k}: {acc:.2f}')

# Make predictions
predicted = model.predict(x_test)

# Print a sample prediction
if len(predicted) > 5:
    print(f'Predicted: {predicted[5]}, Data: {x_test[5]}, Actual: {y_test[5]}')

# Print all predictions
#for i in range(len(predicted)):
 #   print(f'Predicted: {predicted[i]}, Data: {x_test[i]}, Actual: {y_test[i]}')

print(acc)
predicted  = model.predict(x_test)
cm  = confusion_matrix(y_test, predicted)
print(sum(cm))

false_negatives = np.where((y_test == 1) & (predicted == 0))[0]
false_positives = np.where((y_test == 0) & (predicted == 1))[0]

# Inverse transform the standardized data to original scale
x_test_inverse = scaler.inverse_transform(x_test)

# Print the False Negatives and False Positives with actual data
print("False Negatives (Indexes):", false_negatives)
print("False Positives (Indexes):", false_positives)
print("False Negatives (Original Values):", x_test_inverse[false_negatives])
print("False Positives (Original Values):", x_test_inverse[false_positives])
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot= True)
plt.xlabel("pred")
plt.ylabel("trues")
#plt.show()

joblib.dump(model, 'knn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import joblib
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from io import BytesIO
import base64

import sklearn
import sklearn.model_selection
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np 
from sklearn import linear_model,preprocessing
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import joblib
####
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import dash_bootstrap_components as dbc

# Load the model
model = joblib.load('knn_model.pkl')
# Create a Dash app
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, suppress_callback_exceptions=True,external_stylesheets=external_stylesheets)



row_one = html.Div(
    dbc.Row([dbc.Col(html.Div("Age:", style={'font-weight': 'bold'}),width = 6),
             dbc.Col(html.Div("Anaemia:", style={'font-weight': 'bold'}),width = 6)])
)

row_two = html.Br()

row_three = html.Div(
    dbc.Row([dbc.Col(children=[dcc.Input(id='age', type='number', value=50)
        ], width=6),
        dbc.Col(children=[dcc.Input(id='anaemia', type='number', value=0)
        ], width=6)])
    ) 

row_four = html.Div(
    dbc.Row([dbc.Col(children=[dcc.Input(id='creatinine_phosphokinase', type='number', value=50)
        ], width=6),
        dbc.Col(children=[dcc.Input(id='diabetes', type='number', value=0)
        ], width=6)])
    ) 

row_five = html.Div(
    dbc.Row([dbc.Col(children=[dcc.Input(id='ejection_fraction', type='number', value=50)
        ], width=6),
        dbc.Col(children=[dcc.Input(id='high_blood_pressure', type='number', value=0)
        ], width=6)])
    ) 

row_six = html.Div(
    dbc.Row([dbc.Col(children=[dcc.Input(id='platelets', type='number', value=50)
        ], width=6),
        dbc.Col(children=[dcc.Input(id='serum_creatinine', type='number', value=0)
        ], width=6)])
    ) 
row_seven = html.Div(
    dbc.Row([dbc.Col(children=[dcc.Input(id='serum_sodium', type='number', value=50)
        ], width=6),
        dbc.Col(children=[dcc.Input(id='sex', type='number', value=0)
        ], width=6)])
    )

row_eight = html.Div(
    dbc.Row([dbc.Col(children=[dcc.Input(id='smoking', type='number', value=50)
        ], width=6),
        dbc.Col(children=[dcc.Input(id='time', type='number', value=0)
        ], width=6)])
    )

row_nine = html.Div(
    dbc.Row([dbc.Col(children=[], width=3),
        dbc.Col(children=[html.Button('Submit', id='submit-val', n_clicks=0)
        ], width=6),
        dbc.Col(children=[], width=3)])
    ) 

row_ten = html.Div([
    html.H1("Heart Failure Prediction"),])

row_eleven = html.Div([
    html. Div(id='prediction-output')])

row_1 = html.Div(
    dbc.Row([dbc.Col(html.Div("Age:", style={'font-weight': 'bold'}),width= 6),
            dbc.Col(html.Div("Anaemia:", style={'font-weight': 'bold'}),width= 6)])
)
row_2 = html.Div(
    dbc.Row([dbc.Col(html.Div('Creatinine_phosphokinase:', style={'font-weight': 'bold'}),width= 6),
            dbc.Col(html.Div('Diabetes:', style={'font-weight': 'bold'}),width= 6)])
)
row_3 = html.Div(
    dbc.Row([dbc.Col(html.Div('Ejection_fraction:', style={'font-weight': 'bold'}),width= 6),
            dbc.Col(html.Div('High_blood_pressure:', style={'font-weight': 'bold'}),width= 6)])
)
row_4 = html.Div(
    dbc.Row([dbc.Col(html.Div('Platelets:', style={'font-weight': 'bold'}),width= 6),
            dbc.Col(html.Div('Serum_creatinine:', style={'font-weight': 'bold'}),width= 6)])
)
row_5 = html.Div(
    dbc.Row([dbc.Col(html.Div('Serum_sodium:', style={'font-weight': 'bold'}),width= 6),
            dbc.Col(html.Div('Sex:', style={'font-weight': 'bold'}),width= 6)])
)
row_6 = html.Div(
    dbc.Row([dbc.Col(html.Div('Smoking:', style={'font-weight': 'bold'}),width= 6),
            dbc.Col(html.Div('Time:', style={'font-weight': 'bold'}),width= 6)])
)

app.layout = dbc.Container([row_ten,row_two,row_1,row_three,row_two,row_2,row_four,row_two,row_3,row_five,row_two,row_4,row_six,row_two,row_5,row_seven,
                            row_two,row_6,row_eight,row_two,row_nine,row_two,row_eleven
                            ])


@app.callback(
    Output('prediction-output', 'children'),
    Input('submit-val', 'n_clicks'),
    State('age', 'value'), State('anaemia', 'value'), State('creatinine_phosphokinase', 'value'),
     State('diabetes', 'value'), State('ejection_fraction', 'value'), State('high_blood_pressure', 'value'),
     State('platelets', 'value'), State('serum_creatinine', 'value'), State('serum_sodium', 'value'),
     State('sex', 'value'), State('smoking', 'value'), State('time', 'value')
)
def update_output(n_clicks, age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                  high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex, smoking, time):
    if n_clicks > 0:
        features = np.array([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                              high_blood_pressure, platelets, serum_creatinine, serum_sodium,
                              sex, smoking, time]])
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        prediction = model.predict(features_scaled)[0]
        
        # Generate confusion matrix for illustration (using test data)
        y_test_pred = model.predict(x_test)
        #cm_image = generate_confusion_matrix_plot(y_test, y_test_pred)
        
        return f'Prediction: {"Death" if prediction == 1 else "Survival"}', #cm_image
    return '', ''

if __name__ == '__main__':
    app.run_server(debug=True)