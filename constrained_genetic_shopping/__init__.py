import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def constrained_genetic_shopping(df, budget, freshness_days, dietary_restrictions, israeli_product):

    """
    Select products from the DataFrame df within the budget and satisfying dietary restrictions, freshness constraint, and Israeli product constraint.
    """
    selected_products = pd.DataFrame(columns=df.columns)
    remaining_budget = budget

    # Filter products within budget
    available_products = df[df['Price'] <= remaining_budget]

    # Filter products based on dietary restrictions
    if isinstance(dietary_restrictions, dict):
        dietary_restrictions = dietary_restrictions.get("restrictions", "none")

    if dietary_restrictions != "none":
        restrictions_list = [restriction.strip() for restriction in dietary_restrictions.split(',')]
        available_products = available_products[available_products['Dietary Restrictions'].apply(lambda x: any(restriction in x.get('restrictions', []) for restriction in restrictions_list))]

    # Filter products based on freshness constraint
    if freshness_days != 0:
        available_products = available_products[available_products['Days to Expire'] >= int(freshness_days)]

    # Filter products based on Israeli product constraint
    if israeli_product == True:
        available_products = available_products[available_products['Israeli Product'] == "No"]

    # Sort available products by price in Ascending order
    available_products = available_products.sort_values(by='Price', ascending=True)

    # Select products until budget is exhausted
    while remaining_budget > 0 and not available_products.empty:
        # Get the most expensive product
        product = available_products.iloc[0]
        if product['Price'] <= remaining_budget:
            selected_products = pd.concat([selected_products, pd.DataFrame([product])], ignore_index=True)
            remaining_budget -= product['Price']
            available_products = available_products.drop(available_products.index[0])
        else:
            break

    return selected_products

def isHealthy2(selected_products, user_age, total_items):

    max_calories = 500 + (user_age * 5)
    max_fats = 20 + (total_items * 0.5)
    max_carbs = 50 + (total_items * 1)
    max_protein = 30 + (user_age * 2)

    # Calculate total nutritional values of selected products
    total_calories = selected_products['Calories'].sum()
    total_fats = selected_products['Fat'].sum()
    total_carbs = selected_products['Carbs'].sum()
    total_protein = selected_products['Protein'].sum()

    # Initialize the reason variable
    reason = ""

    # Check the reason for unhealthiness
    if total_calories > max_calories:
        reason = "High calorie content"
    elif total_fats > max_fats:
        reason = "High fat content"
    elif total_carbs > max_carbs:
        reason = "High carb content"
    elif total_protein > max_protein:
        reason = "High protein content"
    else:
        reason = None

    # Determine if the combination of selected products is healthy based on thresholds
    is_healthy = (total_calories <= max_calories) and \
                 (total_fats <= max_fats) and \
                 (total_carbs <= max_carbs) and \
                 (total_protein <= max_protein)

    return is_healthy, reason


def train_healthiness_model(df):
    # Assuming you have a dataset with features and labels
    # Features: ['Calories', 'Fat', 'Carbs', 'Protein', 'Age']
    # Label: 'Healthy' (1 if healthy, 0 if unhealthy)

    # Split the data into features and labels
    X = df[['Calories', 'Fat', 'Carbs', 'Proteins', 'Age', 'Items Count']]
    y = df['Healthy']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train a Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predict on the test set and calculate accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)

    return clf

def isHealthy(selected_products, user_age, total_items, healthiness_model):
    # Calculate total nutritional values of selected products
    total_calories = selected_products['Calories'].sum()
    total_fats = selected_products['Fat'].sum()
    total_carbs = selected_products['Carbs'].sum()
    total_protein = selected_products['Protein'].sum()

    # Predict healthiness using the trained model
    prediction = healthiness_model.predict([[total_calories, total_fats, total_carbs, total_protein, user_age, total_items]])

    if prediction == 0:
        reason = "Unhealthy"
    else:
        reason = "Healthy"

    return prediction[0] == 1, reason

def barPlot(selected_products):
    # PLot 1 Bar Plot
    categories_counts = selected_products['Category'].value_counts()
    top_5_categories = categories_counts.head(5)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_5_categories.index, y=top_5_categories.values, palette="viridis")
    plt.title('Top 5 Product Categories')
    plt.xlabel('Category')
    plt.ylabel('Number of Products')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
    plt.grid(True)
    image_file = 'static/Barplot.png'  # Save the image in the static folder
    return plt.savefig(image_file)

def scatterPlot(selected_products):
    plt.figure(figsize=(10, 6))
    plt.scatter(selected_products['Price'], selected_products['Days to Expire'], color='blue', alpha=0.5)
    plt.title('Scatter Plot: Price vs Days to Expire')
    plt.xlabel('Price')
    plt.ylabel('Freshness Days')
    plt.grid(True)
    image_file = 'static/scatterPlot.png'

    return plt.savefig(image_file)

def expiration_date_analysis(df, num_clusters=3):

    if len(df) < num_clusters:
        raise ValueError("Number of samples should be greater than or equal to the number of clusters.")

    df['Days to Expire'].fillna(0)
    df.fillna(0)
    X = df[['Days to Expire']]

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Define a pipeline to handle missing values and perform clustering
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with mean
        ('kmeans', KMeans(n_clusters=num_clusters, random_state=42))
    ])

    # Initialize and fit KMeans clustering model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Create a scatter plot of days until expiration vs. cluster
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Days to Expire'], df['Cluster'], c=df['Cluster'], cmap='viridis')
    plt.xlabel('Days Until Expiration')
    plt.ylabel('Cluster')
    plt.title('Clustering of Products Based on Expiration Date')
    plt.grid(True)
    image_file = 'static/clustering.png'

    # Save the plot as an image file
    plt.savefig(image_file)

    plt.close()

    return 0

def product_segmentation(df, num_clusters=3):

    if len(df) < num_clusters:
        raise ValueError("Number of samples should be greater than or equal to the number of clusters.")

    # Fill missing values
    df.fillna(0, inplace=True)

    # Select features for clustering
    features = ['Calories', 'Fat', 'Carbs', 'Protein']
    X = df[features]

    # Initialize and fit KMeans clustering model
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X)

    # Add cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Plot clustering results
    plt.figure(figsize=(10, 6))
    for cluster in range(num_clusters):
        cluster_data = df[df['Cluster'] == cluster]
        plt.scatter(cluster_data['Calories'], cluster_data['Protein'], label=f'Cluster {cluster}', alpha=0.7)
    plt.xlabel('Calories')
    plt.ylabel('Protein')
    plt.title('Product Segmentation Based on Nutritional Content')
    plt.legend()
    plt.grid(True)
    image_file = 'static/product_segmentation.png'

    # Save the plot as an image file
    plt.savefig(image_file)

    return plt.close()