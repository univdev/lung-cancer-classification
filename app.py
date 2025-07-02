from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# 모델 로드
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'lung_cancer_model.joblib')
    return joblib.load(model_path)

model = load_model()

# 입력 feature 목록 (cancer_preprocessed.csv 참고)
FEATURES = [
    'AGE', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',
    'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
    'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN', 'GENDER_F', 'GENDER_M'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # 폼에서 데이터 추출
        input_data = []
        for feature in FEATURES:
            if feature == 'GENDER_F':
                gender = request.form.get('GENDER')
                value = 1 if gender == 'F' else 0
            elif feature == 'GENDER_M':
                gender = request.form.get('GENDER')
                value = 1 if gender == 'M' else 0
            else:
                value = request.form.get(feature)
                if feature == 'AGE':
                    value = int(value)
                else:
                    value = int(value)  # 0 또는 1
            input_data.append(value)
        # 모델 예측
        X = np.array([input_data])
        pred = model.predict(X)[0]
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            cancer_prob = proba[1] if len(proba) > 1 else None
        else:
            cancer_prob = None
        prediction = {
            'result': '폐암 의심' if pred else '폐암 아님',
            'prob': cancer_prob
        }
    return render_template('index.html', features=FEATURES, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True) 