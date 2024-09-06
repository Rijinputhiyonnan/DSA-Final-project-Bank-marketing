from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import calendar
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

df =pd.read_csv("bank-additional-full.csv")

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
    
    
@app.route('/')

def home():
    return render_template('home.html')



















@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input and convert to appropriate types
        age = int(request.form["age"])
        job = request.form["job"]
        marital = request.form["marital"]
        education = request.form["education"]
        default = request.form["default"]
        housing = request.form["housing"]
        loan = request.form["loan"]
        month_day = request.form["month_day"]
        previous = int(request.form["previous"])
        cons_price_idx = float(request.form["cons.price.idx"])
        euribor3m = float(request.form["euribor3m"])
        duration = int(request.form["duration"])
        campaign = int(request.form["campaign"])
        pdays = int(request.form["pdays"])

        print(f'values entered- {age}, job - {job}, marital - {marital}, education - {education}, default - {default}, housing - {housing}, loan - {loan}, month_day - {month_day}, previous - {previous}, cons_price_idx - {cons_price_idx}, euribor3m - {euribor3m}, duration - {duration}, campaign - {campaign}, pdays - {pdays}')

        # Ensure month_day is in the correct format
        if len(month_day.split('-')) != 3:
            return "Error: month_day should be in 'YYYY-MM-DD' format", 400
        
        # Extract month and day_of_week from 'YYYY-MM-DD'
        year, month, day = month_day.split('-')

        # Validate month and day_of_week
        if not (1 <= int(month) <= 12) or not (1 <= int(day) <= 31):
            return "Error: month should be between 1 and 12, and day should be between 1 and 31", 400
        
        # Convert month to integer
        month = int(month)
        day_of_week = calendar.day_abbr[pd.Timestamp(year=int(year), month=month, day=int(day)).dayofweek].lower()

        # Create DataFrame for input
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'day_of_week': [day_of_week],
            'previous': [previous],
            'cons.price.idx': [cons_price_idx],
            'euribor3m': [euribor3m],
            'month': [calendar.month_abbr[month].lower()],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays]
        })

        # Define the custom order for months
        month_order = ['feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec', 'jan']

        # Assign quartiles and binning after ensuring correct data types
        input_data['month'] = pd.Categorical(input_data['month'], categories=month_order, ordered=True)

        # Assign quartiles based on the month
        input_data['quartile'] = input_data['month'].apply(lambda x: 'Q1' if x in ['feb', 'mar', 'apr'] else
                                                            'Q2' if x in ['may', 'jun', 'jul'] else
                                                            'Q3' if x in ['aug', 'sep', 'oct'] else 'Q4')

        # Binning 'pdays'
        bins_pdays = [-1, 5, 10, 20, float('inf')]
        labels_pdays = ['0-5 days', '6-10 days', '11-20 days', '20+ days']
        input_data['pdays_bin'] = pd.cut(input_data['pdays'], bins=bins_pdays, labels=labels_pdays)

        '''# Binning 'duration'
        bins_duration = [-1, 200, 300, 400, float('inf')]
        labels_duration = ['0-200 sec', '201-300 sec', '301-400 sec', '400+ sec']
        input_data['duration_bin'] = pd.cut(input_data['duration'], bins=bins_duration, labels=labels_duration)'''
        
        # Correct Binning for 'duration' to match the training bins
        bins_duration = [-1, 100, 200, 300, float('inf')]  # Use the correct bins
        labels_duration = ['0-100 sec', '101-200 sec', '200-300 sec', '300+ sec']  # Labels should match the column names in the model
        input_data['duration_bin'] = pd.cut(input_data['duration'], bins=bins_duration, labels=labels_duration)


        # Binning 'campaign'
        bins_campaign = [-1, 5, 10, float('inf')]
        labels_campaign = ['0-5 times', '6-10 times', '10+ times']
        input_data['campaign_bin'] = pd.cut(input_data['campaign'], bins=bins_campaign, labels=labels_campaign)

        # Drop the original columns
        input_data.drop(['month', 'pdays', 'duration', 'campaign'], axis=1, inplace=True)

        # Convert categorical variables to dummy/indicator variables
        input_data = pd.get_dummies(input_data, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'day_of_week', 'pdays_bin', 'duration_bin', 'campaign_bin', 'quartile'])
        required_columns = ['age', 'previous', 'cons.price.idx', 'euribor3m', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid',
                        'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed',
                        'marital_divorced', 'marital_married', 'marital_single', 'education_basic.4y', 'education_basic.6y', 'education_basic.9y',
                        'education_high.school', 'education_illiterate', 'education_professional.course', 'education_university.degree', 'default_no',
                        'default_yes', 'housing_no', 'housing_yes', 'loan_no', 'loan_yes', 'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu',
                        'day_of_week_tue', 'day_of_week_wed', 'pdays_bin_0-5 days', 'pdays_bin_6-10 days', 'pdays_bin_11-20 days', 'pdays_bin_20+ days',
                        'duration_bin_0-100 sec', 'duration_bin_101-200 sec', 'duration_bin_200-300 sec', 'duration_bin_300+ sec', 'campaign_bin_0-5 times',
                        'campaign_bin_6-10 times', 'campaign_bin_10+ times', 'quartile_Q1', 'quartile_Q2', 'quartile_Q3', 'quartile_Q4']

        
        for col in required_columns:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match the trained model
        input_data = input_data[required_columns]


        # Scaling numerical features
        scaler = MinMaxScaler()
        input_data[['age', 'previous', 'cons.price.idx', 'euribor3m']] = scaler.fit_transform(input_data[['age', 'previous', 'cons.price.idx', 'euribor3m']])

        # Ensure all columns are integers
        input_data = input_data.astype(int)

        # Load the model
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)

        # Make prediction
        # After prediction
        pred = model.predict(input_data)
        print(pred, "this is the prediction")

        # Convert the prediction to an integer (if it's in array form)
        pred = int(pred[0])

        # Define the message based on the prediction
        if pred == 1:
            message = "The customer is predicted to subscribe to a term deposit."
        else:
            message = "The customer is predicted not to subscribe to a term deposit."

        # Render the template with the message
        return render_template('home.html', p_result=message)

    except ValueError as e:
        return f"ValueError: {e}", 400
    except Exception as e:
        return f"An error occurred: {e}", 500

if __name__ == '__main__':
    app.run(debug=True)

