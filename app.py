from flask import Flask, render_template, request
import pandas as pd
import pickle

university_data = pd.read_csv('university_data.csv')

model = pickle.load(open('admission_model.pkl', 'rb'))

features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA' , 'Research']

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    # Access form data
    gre_score = int(request.form['gre_score'])
    toefl_score = int(request.form['toefl_score'])
    university_rating = int(request.form['university_rating'])
    sop = float(request.form['sop'])
    lor = float(request.form['lor'])
    cgpa = float(request.form['cgpa'])
    research_string = request.form['research']
    if research_string == 'yes':
        research = 1
    else:
        research = 0
    
    # Create input DataFrame
    input_data = [[
        gre_score,
        toefl_score,
        university_rating,
        sop,
        lor,
        cgpa,
        research
        
    ]]
    
    # Make prediction
    prediction = model.predict(input_data)[0] * 100
    prediction = round(prediction,2)
    if prediction > 100:
        prediction = 100

    # Filter universities based on prediction percentage
    filtered_universities = university_data[university_data['Percentage'] <= float(prediction)].reset_index(drop=True)

    # Sort universities in descending order based on percentage
    sorted_universities = filtered_universities.sort_values(by='Percentage', ascending=False)

    return render_template('result.html', prediction=prediction, universities=sorted_universities.to_html(index=False))

if __name__ == '__main__':
    app.run(debug=True)