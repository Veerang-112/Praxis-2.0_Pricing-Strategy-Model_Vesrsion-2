import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_processor import DataProcessor
import numpy as np

def visualize_relationships():
    # Load and process data
    file_path = r"c:\Users\DELL\OneDrive\Desktop\anti\pricing_dataset.csv"
    print("Loading data...")
    processor = DataProcessor(file_path)
    df = processor.process()
    
    if df is None:
        print("Failed to load data.")
        return

    # Prepare data for plotting
    # Helper to handle log scale for Demand (Rating Count) + 1 to avoid log(0)
    df['log_demand'] = np.log1p(df['rating_count'])
    
    # Set up the figure with subplots using constrained layout for better spacing
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout='constrained')
    fig.suptitle('Key Relationships in Pricing Data (with Regression Lines)', fontsize=16)
    
    # 1. Price vs Demand (Elasticity)
    print("Plotting Price vs Demand...")
    sns.regplot(ax=axes[0, 0], data=df, x='discounted_price', y='log_demand', 
                scatter_kws={'alpha':0.3}, line_kws={'color':'red'})
    axes[0, 0].set_title('Price vs Demand (Log Scale)')
    axes[0, 0].set_xlabel('Discounted Price (₹)')
    axes[0, 0].set_ylabel('Log(Rating Count)')
    axes[0, 0].set_xscale('log') # Log scale for price too usually helps viz

    # 2. Discount % vs Demand
    print("Plotting Discount % vs Demand...")
    sns.regplot(ax=axes[0, 1], data=df, x='discount_percentage', y='log_demand', 
                scatter_kws={'alpha':0.3}, line_kws={'color':'green'})
    axes[0, 1].set_title('Discount % vs Demand (Log Scale)')
    axes[0, 1].set_xlabel('Discount Percentage (0.0 - 1.0)')
    axes[0, 1].set_ylabel('Log(Rating Count)')

    # 3. Rating vs Demand
    print("Plotting Rating vs Demand...")
    sns.regplot(ax=axes[1, 0], data=df, x='rating', y='log_demand', 
                scatter_kws={'alpha':0.3}, line_kws={'color':'orange'})
    axes[1, 0].set_title('Product Rating vs Demand (Log Scale)')
    axes[1, 0].set_xlabel('Rating (0-5)')
    axes[1, 0].set_ylabel('Log(Rating Count)')

    # 4. Sentiment vs Demand
    print("Plotting Sentiment vs Demand...")
    sns.regplot(ax=axes[1, 1], data=df, x='sentiment_score', y='log_demand', 
                scatter_kws={'alpha':0.3}, line_kws={'color':'purple'})
    axes[1, 1].set_title('Review Sentiment vs Demand (Log Scale)')
    axes[1, 1].set_xlabel('Sentiment Score (-1 to 1)')
    axes[1, 1].set_ylabel('Log(Rating Count)')

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Replaced by layout='constrained'
    
    print("Displaying plots...")
    plt.show()

if __name__ == "__main__":
    visualize_relationships()
