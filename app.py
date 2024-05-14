from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import datetime
import constrained_genetic_shopping as cons

from constrained_genetic_shopping import isHealthy

import seaborn as sns
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = 'secret_key'

# Load the products DataFrame
df = pd.read_csv("items.csv")

user_item_interactions = pd.read_csv("user_item_interactions.csv")


exchange_rate = 278

df['Total Price'] = df['Price'] * exchange_rate
df['Price'] = df['Total Price'] / df['Quantity']
df['Price'] = df['Price'].round(2)

df['Expiration Date'] = pd.to_datetime(df['Expiration Date'])


def calculate_days_to_expire(expiration_date):
    if pd.isnull(expiration_date):
        return None
    today = pd.Timestamp.now().normalize()
    days_to_expire = (expiration_date - today).days
    return days_to_expire

df['Days to Expire'] = df['Expiration Date'].apply(calculate_days_to_expire)
df['Days to Expire'].fillna(0)

#converting to dictionary
df['Dietary Restrictions'] = df['Dietary Restrictions'].apply(lambda x: {'restrictions': [x]} if pd.notnull(x) else {'restrictions': []})


startTime = datetime.datetime.now()
# Flask routes
@app.route('/')
def index():
    return render_template('index.html', items=df['Product Name'].tolist(), df=df)


@app.route('/results', methods=['GET','POST'])
def results():
    global user_item_interactions  # Declare user_item_interactions as a global variable

    user_id = request.form.get('userID')
    gender = request.form.get('gender')
    age = int(request.form.get('age'))

    # Get user inputs from the form
    selected_items_list = request.form.get('selected_items_list')  # Retrieve the selected items list
    selected_items = [item.strip() for item in selected_items_list.split(',') if item.strip()]  # Convert selected items string to list
    selected_df = df[df['Product Name'].isin(selected_items)]  # Filter the DataFrame based on selected item names

    budget = float(request.form['budget'])
    freshness_days = int(request.form['freshness_days'])
    dietary_restrictions = request.form['dietary_restrictions']
    israeli_product = request.form.get('israeli_product') == 'Yes'

    # Call your function with user inputs
    selected_products = cons.constrained_genetic_shopping(selected_df, budget, freshness_days, dietary_restrictions, israeli_product)

    # Calculate total spend
    total_spend = selected_products['Price'].sum().__float__().__round__(2)

    # Get the list of selected products
    selected_products_ids = selected_products['Product ID'].tolist()

    # Extract user ID (assuming you have a way to identify users)
    endTime = datetime.datetime.now()
    session_duration = endTime-startTime
    session_duration = session_duration.total_seconds()

    items_count = len(selected_products)  # Total number of items

    # Calculate the count of items for each category in the selected products DataFrame
    category_counts = selected_products['Category'].value_counts().to_dict()

    #Healthy
    healthiness_model = cons.train_healthiness_model(user_item_interactions)
    #healthy, reason = isHealthy(user_item_interactions, age, items_count, healthiness_model)
    healthy, reason = cons.isHealthy2(user_item_interactions, age, items_count)

    # Extract 'Calories', 'Fat', 'Carbs', 'Protein' values
    nutritional_values = selected_products[['Calories', 'Fat', 'Carbs', 'Protein']].sum()

    # Create a row to save data in a new .csv file for Machine Learning
    new_row = pd.DataFrame({
        "user ids": user_id,
        "Gender" : [gender],
        "Age": age,
        "selected items": [selected_products_ids],
        "Items Count": items_count,
        "Budget": budget,
        "Spend": total_spend,
        "Session Duration": session_duration,
        "Healthy": healthy,
        "Calories": nutritional_values['Calories'],
        "Fat": nutritional_values['Fat'],
        "Carbs": nutritional_values['Carbs'],
        "Proteins": nutritional_values['Protein'],
        **category_counts
    })

    user_item_interactions = pd.concat([user_item_interactions, new_row], ignore_index=True)

    # Save the updated user-item interactions DataFrame
    user_item_interactions.to_csv("user_item_interactions.csv", index=False)

    #Matplotlib
    try:
        # Plot 1: Bar Plot
        cons.barPlot(selected_products)
    except Exception as e:
        print("Error occurred while creating the bar plot:", str(e))

    try:
        # Plot 2: Scatter Plot
        cons.scatterPlot(selected_products)
    except Exception as e:
        print("Error occurred while creating the scatter plot:", str(e))

    try:
        # Plot 3: Expiration Date Analysis
        cons.expiration_date_analysis(selected_products)
    except Exception as e:
        print("Error occurred while performing expiration date analysis:", str(e))

    try:
        # Plot 4: Expiration Date Analysis
        cons.product_segmentation(selected_products)
    except Exception as e:
        print("Error occurred while performing Product Segmentation:", str(e))

    # Convert selected_products to HTML table for rendering
    table_html = selected_products.to_html(index=False)

    if not healthy:
        return render_template('results.html', table_html=table_html, total_spend=total_spend, unhealthy_reason=reason)
    else:
        return render_template('results.html', table_html=table_html, total_spend=total_spend)

@app.route('/analysis_popup.html')
def analysis_popup():
    return render_template('analysis_popup.html')


if __name__ == '__main__':
    app.run(debug=True)