from django.shortcuts import render
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

model = joblib.load("lr_model.pkl")

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    feature1 = float(request.GET['n1'])
    feature2 = float(request.GET['n2'])
    feature3 = float(request.GET['n3'])
    feature4 = float(request.GET['n4'])
    feature5 = float(request.GET['n5'])
    feature6 = float(request.GET['n6'])
    feature7 = float(request.GET['n7'])
    feature8 = float(request.GET['n8'])

    user_input = pd.DataFrame({
        'feature1': [feature1],
        'feature2': [feature2],
        'feature3': [feature3],
        'feature4': [feature4],
        'feature5': [feature5],
        'feature6': [feature6],
        'feature7': [feature7],
        'feature8': [feature8],
    })

    scaler = StandardScaler()
    user_input_scaled = scaler.fit_transform(user_input)

    prediction = model.predict(user_input_scaled)
    print(prediction)
    result1 = ""
    if prediction == [1]:
        result1 = "Positive"
    elif prediction == [0]:
        result1 = "Negative"

    return render(request, "predict.html", {"result2":result1})