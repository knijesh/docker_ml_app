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
    
    print("Printing Request Data")
    print(request_data)
    
    values = request_data['values']
    fields = request_data['fields']
    
    data = pd.DataFrame(values,columns=fields)
   
    with open('pipeline-model.pkl', 'rb') as inp:
        import os
        mod = pickle.load(inp)
    
    fields = ['Count', 'Country', 'State', 'City', 'Latitude', 'Longitude', 'Partner', 'Dependents', 
              'TenureMonths', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
              'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
              'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges'] 
    
    test_data = data
    X_test = test_data[fields]
    
    y_pred =  mod.predict(X_test).tolist()
    y_prob =  mod.predict_proba(X_test).tolist()

    new_fields = fields.copy()

    fields.append('prediction')
    fields.append('probability')

    values = test_data.values.tolist()
            
    i = 0
    while i < len(y_pred) :
        values[i].append(y_pred[i])
        i +=1

    j=0  
    while j < len(y_prob) :
        values[j].append(y_prob[j])
        j +=1
        
        
    format = {
        
        "fields":fields,
        "labels":[1,0],
        "values":values  
    }
    
    
    return format

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)