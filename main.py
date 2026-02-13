from data_processor import DataProcessor
from demand_model import DemandModel
from pricing_scenario import PricingScenario
import random
import os

def main():
    file_path = r"c:\Users\DELL\OneDrive\Desktop\anti\pricing_dataset.csv"
    model_path = r"c:\Users\DELL\OneDrive\Desktop\anti\pricing_model.pkl"
    
    print("Initializing Data Processor...")
    processor = DataProcessor(file_path)
    
    # Process Data
    df = processor.process()
    
    if df is not None:
        print("\nDATA STATISTICS:")
        print(df.describe())
        
        print("\n--- Training Model ---")
        model = DemandModel()
        score = model.train(df)
        
        print(f"\nModel Training Complete. R^2 Score: {score:.4f}")
        
        # Save model
        model.save(model_path)
        
        # --- SCENARIO EXPLORATION DEMO ---
        print("\n--- Starting Pricing Scenario Exploration ---")
        scenario = PricingScenario(model_path, df)
        
        # 1. Feature Importance
        print("\n[Feature Importance]")
        importance_df = scenario.get_feature_importance()
        if importance_df is not None:
            print(importance_df.head(10))
            
        # 2. Simulate Price Elasticity for a Random Product
        print("\n[Price Elasticity Simulation]")
        # Pick a random product that has a reasonable price
        sample_product = df.sample(1)
        product_id = sample_product['product_id'].values[0]
        product_name = sample_product['product_name'].values[0]
        
        print(f"Selected Product: {product_name[:50]}... (ID: {product_id})")
        
        results = scenario.simulate_price_elasticity(sample_product, price_range_pct=0.2, steps=5)
        print("\nSimulation Results (Price vs Demand vs Revenue):")
        print(results)
        
        # 3. Find Optimal Price (simple search)
        print("\n[Optimal Price Search]")
        scenario.find_optimal_price(product_id)
        
    else:
        print("Data processing failed.")

if __name__ == "__main__":
    main()
