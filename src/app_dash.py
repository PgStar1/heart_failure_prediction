
import dash
import dash_core_components as dcc
import dash_html_components as html
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


# Load the model
model = joblib.load('knn_model.pkl')

# Function to generate confusion matrix plot
def generate_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7,5))
    sn.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    img = base64.b64encode(buf.getvalue()).decode('utf-8')
    return 'data:image/png;base64,{}'.format(img)

# Create a Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Heart Failure Prediction"),
    html.Div([
        html.Label('Age'),
        dcc.Input(id='age', type='number', value=50),
        html.Label('Anaemia'),
        dcc.Input(id='anaemia', type='number', value=0),
        html.Label('Creatinine Phosphokinase'),
        dcc.Input(id='creatinine_phosphokinase', type='number', value=250),
        html.Label('Diabetes'),
        dcc.Input(id='diabetes', type='number', value=0),
        html.Label('Ejection Fraction'),
        dcc.Input(id='ejection_fraction', type='number', value=35),
        html.Label('High Blood Pressure'),
        dcc.Input(id='high_blood_pressure', type='number', value=0),
        html.Label('Platelets'),
        dcc.Input(id='platelets', type='number', value=250000),
        html.Label('Serum Creatinine'),
        dcc.Input(id='serum_creatinine', type='number', value=1.2),
        html.Label('Serum Sodium'),
        dcc.Input(id='serum_sodium', type='number', value=135),
        html.Label('Sex'),
        dcc.Input(id='sex', type='number', value=1),
        html.Label('Smoking'),
        dcc.Input(id='smoking', type='number', value=0),
        html.Label('Time'),
        dcc.Input(id='time', type='number', value=100),
        html.Button('Submit', id='submit-val', n_clicks=0)
    ]),
    html.Div(id='prediction-output'),
    html.Img(id='confusion-matrix')
])

@app.callback(
    [Output('prediction-output', 'children'),
     Output('confusion-matrix', 'src')],
    [Input('submit-val', 'n_clicks')],
    [State('age', 'value'), State('anaemia', 'value'), State('creatinine_phosphokinase', 'value'),
     State('diabetes', 'value'), State('ejection_fraction', 'value'), State('high_blood_pressure', 'value'),
     State('platelets', 'value'), State('serum_creatinine', 'value'), State('serum_sodium', 'value'),
     State('sex', 'value'), State('smoking', 'value'), State('time', 'value')]
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
        cm_image = generate_confusion_matrix_plot(y_test, y_test_pred)
        
        return f'Prediction: {"Death" if prediction == 1 else "Survival"}', cm_image
    return '', ''

if __name__ == '__main__':
    app.run_server(debug=True)
