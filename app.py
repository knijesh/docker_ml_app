from flask import Flask, render_template,request
import os
app = Flask(__name__)


@app.route('/')
def fun():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def prediction():

    import pandas as pd
    import pickle
    import warnings
    warnings.filterwarnings("ignore")
    
    request_data = request.get_json()
    
    print(request_data)
    
    values = request_data['input_data'][0]['values']
    fields = request_data['input_data'][0]['fields']
    
    data = pd.DataFrame(values,columns=fields)
   
    
    with open('pipeline-model.pkl', 'rb') as inp:
        import os
        print(os.getcwd())
        print(os.listdir())
        mod = pickle.load(inp)
    
    fields = ['Count', 'Country', 'State', 'City', 'Latitude', 'Longitude', 'Partner', 'Dependents', 
              'TenureMonths', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
              'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
              'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'] 
    
    test_data = data
    X_test = test_data[fields]
    
    #y_test = test_data['ChurnValue']
    
    prediction=mod.predict(X_test)
    #print(prediction)
    
    y_pred =  mod.predict(X_test).tolist()
    y_prob =  mod.predict_proba(X_test).tolist()

    vals = list(zip(y_pred,y_prob))
    
    format = {
        "predictions": [
            {
                "fields": [
                    "prediction",
                    "probability"
                ],
                "values": vals
            }
        ]
    }
    
    
    return format




if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)