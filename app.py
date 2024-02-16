from flask import Flask , render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('./cleaned_data.csv')
pipe = pickle.load(open("model.pkl","rb"))

@app.route('/')
def index():
    regions = sorted(data['region'].unique())
    types = sorted(data['type'].unique())
    statuss = sorted(data['status'].unique())
    
    return render_template('index.html', regions=regions, types=types, statuss=statuss)

@app.route('/predict', methods=['POST'])
def predict():
    bhk = request.form.get('bhk')
    type= request.form.get('type')
    area= request.form.get('area')
    region= request.form.get('region')
    status= request.form.get('status')
    
    
    print(bhk,type,area,region,status)
    input = pd.DataFrame([[bhk,type,area,region,status]],columns=['bhk','type','area','region','status'])
    prediction=pipe.predict(input)[0]
    
     # Format prediction as a string with currency symbol
    formatted_prediction = f'₹ {prediction:.2f} Lacs'
    
    return render_template('result.html', prediction=formatted_prediction)

if __name__=='__main__':
    app.run(debug=True)