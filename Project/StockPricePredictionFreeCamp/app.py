
from logging import debug
from flask import Flask, render_template , request
import utils

app = Flask(__name__)  

@app.route('/') 
def home(): 
    return render_template('index.html') 

@app.route('/predict', methods=['POST' , 'GET'])
def predict():
    if request.method == 'POST':
        Open = request.form.get('Open')
        High = request.form.get('High')
        Low = request.form.get('Low')
        Volume = request.form.get('Volume')
    
        prediction = utils.predict_price(Open, High, Low, Volume)

        return render_template("prediction.html", prediction=prediction)

if __name__ == "__main__": 
    app.run(debug=True) 