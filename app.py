from flask import Flask , request , jsonify , render_template
import numpy as np
from sklearn.metrics import accuracy_score
import joblib

# Create flask app
app = Flask(__name__,template_folder='Templates')

# Load pickle model
model = joblib.load("model1.joblib")
scaler = joblib.load('scaler1.joblib')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/result', methods=["POST"])
def result(): 
    # تحويل البيانات المدخلة إلى أرقام عددية
    hemo = float(request.form['hemo'])
    nasb = float(request.form['nasb'])
    age = int(request.form['age'])
    bmi = int(request.form['bmi'])
    sex = int(request.form['sex'])
    haml = int(request.form['haml'])
    smoke = int(request.form['smoke'])
    activite = int(request.form['activite'])
    amla7 = int(request.form['amla7'])
    k7owl = float(request.form['k7owl'])
    aghad = int(request.form['aghad'])
    fshl = int(request.form['fshl'])
    goda = int(request.form['8oda'])

    #  بتحويل البيانات إلى مصفوفة numpy
    data = np.array([hemo, nasb, age, bmi, sex, haml,smoke, activite, amla7, k7owl, aghad, fshl, goda]).reshape(1, -1)
    
    #  بتحويل البيانات باستخدام المقياس المدرب
    vect = scaler.transform(data)
    
    # توقع النموذج على البيانات
    model_prediction = model.predict(vect)
    
    return render_template('index.html', label=model_prediction)


if __name__ == '__main__':
	app.run()
    