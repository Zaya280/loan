import os
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Загварыг pickle-аар унших эсвэл сургалтын модулаас дахин авах логик
model_file = 'model.pkl'
if os.path.exists(model_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print("Модель файл уншигдлаа.")
else:
    # Хэрэв model.pkl байхгүй бол train_model.py–д загвараа сургасан гэж үзээд, 
    # үүнийг импортлох эсвэл өөрийн кодоо оруулах боломжтой.
    # Доор зөвхөн тайлбар зорилгоор бичсэн.
    from train_model import best_model  # train_model.py–д best_model-ыг export хийсэн гэж үзье.
    model = best_model
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print("Модель сургаж, pickle-д хадгаллаа.")

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    # Жишээ: {'gender': 'Male', 'age': '30', 'income': '50000', 'credit_score': '650', 'loan_amount': '150000'}
    gender_str   = form_data.get('gender')
    age          = float(form_data.get('age', 0))
    income       = float(form_data.get('income', 0))
    loan_amount  = float(form_data.get('loan_amount', 0))
    
    # Gender утгыг хөрвүүлэх (Male = 1, Female = 0)
    if gender_str == 'Male':
        gender_val = 1
    else:
        gender_val = 0

    # Загвар сурсан үед ашигласан дарааллаар 5 утгыг NumPy array-д бүрдүүлнэ
    input_features = np.array([gender_val, age, income, loan_amount], dtype=float).reshape(1, -1)
    
    # Таамаглал хийх
    pred_class = model.predict(input_features)[0]
    if pred_class == 1:
        result_text = "Approved"
    else:
        result_text = "Not Approved"

    # Үр дүнг HTML хуудас руу дамжуулах
    test_accuracy = 0.85  # Жишээ утга; бодит үнэлгээгээр солих боломжтой
    class_report = "Precision: 0.80, Recall: 0.82, F1-Score: 0.81"  # Жишээ тайлан

    return render_template('result.html', prediction=result_text, accuracy=test_accuracy, report=class_report)

if __name__ == "__main__":
    app.run(debug=True)
