import pandas as pd
from pycaret.classification import predict_model, load_model

def load_data(filepath):
    df = pd.read_csv(filepath, index_col='customerID')
    return df

"""
This function preprocesses the data in the same way that that my assignment from week 2 does
in order to have the new data in the proper format.
"""
def preprocess_data(df):
    df.fillna(0, inplace=True)

    # Add a row with bank transfer payment in order to have all the dummy variables
    df.loc[123] = {'PaymentMethod': 'Bank transfer (automatic)'}
    df = pd.get_dummies(data=df, drop_first=True, dtype=int)

    # Drop the added row
    df.dropna(inplace=True)

    df['total_charges_tenure_ratio'] = df['TotalCharges']/df['tenure']
    df['monthly_charges_times_tenure'] = df['MonthlyCharges']*df['tenure']
    df.fillna(0, inplace=True)
    
    return df

def make_predictions(df):
    # Loade the model
    model = load_model('QDA')

    # Make the predictions
    predictions = predict_model(model, data=df)

    # Rename the labels
    predictions.rename({'Label': 'Churn_predictions'}, axis=1, inplace=True)

    # Get the prediction probabilities and put them in a dataframe
    prediction_probs = model.predict_proba(df)[:,1]
    prob_df = pd.DataFrame(prediction_probs, index=df.index, columns=['Churn_probability'])

    # Add churn prediction percentile and concat to predictions dataframe
    prob_df['Churn_percentile'] = prob_df['Churn_probability'].rank(pct=True)
    result_df = pd.concat([predictions, prob_df], axis=1)

    # Return predictions, probabilities, and percentiles
    return result_df[['Churn_predictions', 'Churn_probability', 'Churn_percentile']]

if __name__ == "__main__":
    df = load_data('data/new_churn_data_unmodified.csv')
    df = preprocess_data(df)
    print(make_predictions(df))
