import pandas as pd
import pickle

from flask                     import Flask, request, Response
from costadeldata.CostaDelData import CostaDelData

model = pickle.load(open('model/xgb_classifier.pkl', 'rb'))

app = Flask[__name__]

@app.route('/costadeldata/predict', methods=['POST'])
def costadeldata_predict():
    test_json = request.get_json()
    
    if test_json: # test_json tem dados
        if isinstance(test_json, dict): # exemplo único
            test_raw = pd.DataFrame(test_json, index=[0])
            
        else: # múltiplos exemplos
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
        
        # instanciando a Classe CostaDelData
        pipeline = CostaDelData()
        
        # limpeza dos dados
        df1 = pipeline.data_cleaning(test_raw)
        
        # preparação dos dados
        df2 = pipeline.data_preparation(df1)
        
        # predição
        df_response = pipeline.get_prediction(model = model, original_data = test_raw, test_data = df2)
        
        return df_response
    else:
        return Response('{}', status=200, mimetype='application/json')

if __name__ == '__main__':
    app.run('0.0.0.0')