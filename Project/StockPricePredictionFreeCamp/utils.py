import joblib
import numpy as np

def predict_price(Open , High , Low , Volume):
    test_data = np.array([[Open , High , Low , Volume]])
    trainedModel = joblib.load('D:\MLLearning\Project\StockPricePredictionFreeCamp\stock_prediction.pkl')
    prediction = trainedModel.predict(test_data)
    return prediction