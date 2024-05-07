import joblib

model = joblib.load('D:\MLLearning\Project\StockPricePredictionFreeCamp\stock_prediction.pkl')

Open = 208.8
High = 213.4
Low = 206.25
Volume = 2156024

prediction = model.predict([[Open , High , Low , Volume]])

print(prediction)