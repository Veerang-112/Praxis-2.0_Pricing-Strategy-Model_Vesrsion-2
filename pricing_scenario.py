import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from demand_model import DemandModel

class PricingScenario:
    def __init__(self, model_path, data):
        """
        model_path: Path to the trained .pkl file
        data: DataFrame containing the product data (for sampling)
        """
        self.model_loader = DemandModel()
        self.model_loader.load(model_path)
        self.pipeline = self.model_loader.pipeline
        self.data = data

    def get_feature_importance(self):
        """Extracts feature importance from the Random Forest model."""
        try:
            # Access the regressor from the pipeline
            rf_model = self.pipeline.named_steps['regressor']
            
            # Get feature names from preprocessor
            preprocessor = self.pipeline.named_steps['preprocessor']
            
            # Numeric features are passed through
            num_features = ['discounted_price', 'actual_price', 'discount_percentage', 'rating', 'sentiment_score']
            
            # Categorical features are one-hot encoded
            cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(['category_main'])
            
            all_features = num_features + list(cat_features)
            
            importances = rf_model.feature_importances_
            
            # Create DataFrame
            feature_imp_df = pd.DataFrame({'Feature': all_features, 'Importance': importances})
            return feature_imp_df.sort_values(by='Importance', ascending=False)
        except Exception as e:
            print(f"Error extracting feature importance: {e}")
            return None

    def simulate_price_elasticity(self, product_row, price_range_pct=0.2, steps=10):
        """
        Simulates demand for a specific product by varying its price.
        product_row: A single row DataFrame representing the product.
        price_range_pct: The percentage to vary price (e.g., 0.2 for +/- 20%)
        """
        results = []
        base_price = product_row['discounted_price'].values[0]
        
        # Create price points
        prices = np.linspace(base_price * (1 - price_range_pct), base_price * (1 + price_range_pct), steps)
        
        print(f"\n--- Simulation for Product: {product_row['product_name'].values[0][:50]}... ---")
        print(f"Base Price: ₹{base_price}")
        
        for p in prices:
            # Create a copy of the row with new price
            sim_row = product_row.copy()
            sim_row['discounted_price'] = p
            # Update discount percentage assuming actual_price stays constant
            sim_row['discount_percentage'] = (sim_row['actual_price'] - p) / sim_row['actual_price']
            
            # Predict demand
            predicted_demand = self.model_loader.predict(sim_row)[0]
            
            # Calculate Revenue
            revenue = p * predicted_demand
            
            results.append({
                'Price': p,
                'Predicted_Demand': predicted_demand,
                'Revenue': revenue
            })
            
        results_df = pd.DataFrame(results)
        return results_df

    def find_optimal_price(self, product_id):
        """Finds the price that maximizes revenue for a given product ID."""
        product_row = self.data[self.data['product_id'] == product_id]
        if product_row.empty:
            print("Product not found.")
            return None
            
        simulation_df = self.simulate_price_elasticity(product_row)
        
        best_scenario = simulation_df.loc[simulation_df['Revenue'].idxmax()]
        
        print(f"\nOptimal Price: ₹{best_scenario['Price']:.2f}")
        print(f"Projected Revenue: ₹{best_scenario['Revenue']:.2f}")
        print(f"Projected Demand: {best_scenario['Predicted_Demand']:.2f}")
        
        return best_scenario, simulation_df
