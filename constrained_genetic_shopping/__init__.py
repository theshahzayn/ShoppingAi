import pandas as pd


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