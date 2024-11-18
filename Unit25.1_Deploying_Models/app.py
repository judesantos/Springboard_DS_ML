from flask import Flask, request, jsonify
import joblib
import traceback

import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    return _predict()

def _predict():
    if lr:
        try:
            # Predict input using our model
            json_ = request.json
            query_df = pd.DataFrame(json_)
            query = pd.get_dummies(query_df).reindex(columns=model_columns, fill_value=0)
            prediction = lr.predict(query)

            return jsonify({'prediction': str(list(prediction))})
        except:
            return jsonify({'error': traceback.format_exc()})

    else:
        print('Model not found!')
        return ('Can not process request: No model found.')


if __name__ == '__main__':

    # Initialize: load our model
    lr = joblib.load('linear_regression_model.pkl')
    model_columns = joblib.load('lr_model_columns.pkl')

    app.run(debug=True, port=5001)


