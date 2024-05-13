from flask import Flask, render_template, request, session
import pandas as pd
import datetime

import constrained_genetic_shopping as cons

app = Flask(__name__)

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

df['Dietary Restrictions'] = df['Dietary Restrictions'].apply(lambda x: {'restrictions': [x]} if pd.notnull(x) else {'restrictions': []})



startTime = datetime.datetime.now()
# Flask routes
@app.route('/')
def index():
    return render_template('index.html', items=df['Product Name'].tolist(), df=df)

@app.route('/results', methods=['POST'])

def results():

    global user_item_interactions  # Declare user_item_interactions as a global variable


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
    user_id = session.get('user_id')
    endTime = datetime.datetime.now()
    session_duration = endTime-startTime
    session_duration = session_duration.total_seconds()

    items_count = len(selected_products)  # Total number of items

    # Calculate the count of items for each category in the selected products DataFrame
    category_counts = selected_products['Category'].value_counts().to_dict()

    # Create a row to save data in a new .csv file for Machine Learning
    new_row = pd.DataFrame({
        "user ids": [user_id],
        "selected items": [selected_products_ids],
        "Items Count": items_count,
        "Budget": budget,
        "Spend": total_spend,
        "Session Duration": session_duration,
        **category_counts
    })

    user_item_interactions = pd.concat([user_item_interactions, new_row], ignore_index=True)

    # Save the updated user-item interactions DataFrame
    user_item_interactions.to_csv("user_item_interactions.csv", index=False)

    # Convert selected_products to HTML table for rendering
    table_html = selected_products.to_html(index=False)

    return render_template('results.html', table_html=table_html, total_spend=total_spend)

if __name__ == '__main__':
    app.run(debug=True)