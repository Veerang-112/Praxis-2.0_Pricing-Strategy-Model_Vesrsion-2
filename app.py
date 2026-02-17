import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor
from demand_model import DemandModel

# Set page config
st.set_page_config(page_title="Pricing Strategy & Demand Simulator", layout="wide")

# Custom CSS for dropdown behavior
st.markdown("""
<style>
/* Change cursor to pointer for dropdowns */
div[data-testid="stSelectbox"] > div > div {
    cursor: pointer !important;
}

/* Disable typing in the dropdown input by making it read-only-like and transparent caret */
div[data-testid="stSelectbox"] input {
    cursor: pointer !important;
    caret-color: transparent !important;
}

/* Ensure the dropdown arrow also shows pointer */
div[data-testid="stSelectbox"] svg {
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

# Paths
DATA_PATH = r"c:\Users\DELL\OneDrive\Desktop\anti\pricing_dataset.csv"
MODEL_PATH = r"c:\Users\DELL\OneDrive\Desktop\anti\pricing_model.pkl"

@st.cache_data
def load_data():
    processor = DataProcessor(DATA_PATH)
    return processor.process()

@st.cache_resource
def load_model():
    model = DemandModel()
    model.load(MODEL_PATH)
    return model

def main():
    st.title("🛒 Pricing Strategy & Demand Exploration")
    st.markdown("### Retail Analytics Hackathon Solution")
    
    # Load resources
    with st.spinner("Loading Data and Model..."):
        try:
            df = load_data()
            model = load_model()
        except Exception as e:
            st.error(f"Error loading resources: {e}")
            return

    # Sidebar: Product Selection
    st.sidebar.header("1. Select Product")
    
    # Create a label for the dropdown (Name + ID)
    df['label'] = df['product_name'].apply(lambda x: x[:50] + "...") + " (" + df['product_id'] + ")"
    product_label = st.sidebar.selectbox("Choose a product to analyze:", df['label'].unique())
    
    # Get selected product data
    product_id = product_label.split("(")[-1].strip(")")
    product_row = df[df['product_id'] == product_id].iloc[0]
    
    # Sidebar: Scenario Inputs
    st.sidebar.header("2. Simulation Parameters")
    st.sidebar.markdown("Adjust these to see impact on demand.")
    
    current_price = product_row['discounted_price']
    current_rating = product_row['rating']
    current_sentiment = product_row['sentiment_score']
    
    # Sliders
    new_price = st.sidebar.slider("Set Price (₹)", 
                                  min_value=float(max(10, current_price * 0.5)), 
                                  max_value=float(current_price * 2.0), 
                                  value=float(current_price),
                                  step=10.0)
    
    new_rating = st.sidebar.slider("Target Rating (0-5)", 
                                   min_value=1.0, max_value=5.0, 
                                   value=float(current_rating), step=0.1)
                                   
    new_sentiment = st.sidebar.slider("Sentiment Score (-1 to 1)", 
                                      min_value=-1.0, max_value=1.0, 
                                      value=float(current_sentiment), step=0.1)

    # --- SIMULATION ---
    # Prepare simulation row
    sim_row = product_row.to_frame().T
    sim_row['discounted_price'] = new_price
    sim_row['rating'] = new_rating
    sim_row['sentiment_score'] = new_sentiment
    # Recalculate discount % based on fixed ACTUAL price
    sim_row['discount_percentage'] = (sim_row['actual_price'] - new_price) / sim_row['actual_price']
    
    # Predictions
    base_demand = model.predict(product_row.to_frame().T)[0]
    new_demand = model.predict(sim_row)[0]
    
    base_revenue = current_price * base_demand
    new_revenue = new_price * new_demand
    
    # --- DASHBOARD LAYOUT ---
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Product Details")
        st.write(f"**Name:** {product_row['product_name'][:100]}...")
        st.write(f"**Category:** {product_row['category_main']}")
        st.write(f"**Actual Price:** ₹{product_row['actual_price']}")
        st.metric("Current Price", f"₹{current_price}")
        st.metric("Current Rating", f"{current_rating} ⭐")
        
        st.markdown("---")
        st.subheader("Simulation Results")
        
        d_delta = new_demand - base_demand
        r_delta = new_revenue - base_revenue
        
        st.metric("Predicted Demand (Units)", f"{int(new_demand)}", f"{int(d_delta)}")
        st.metric("Projected Revenue", f"₹{new_revenue:,.2f}", f"₹{r_delta:,.2f}")

    with col2:
        st.subheader("Sensitivity Analysis: Price Elasticity")
        
        # Generate Curve Data
        prices = np.linspace(current_price * 0.5, current_price * 1.5, 20)
        demands = []
        revenues = []
        
        for p in prices:
            temp_row = sim_row.copy()
            temp_row['discounted_price'] = p
            temp_row['discount_percentage'] = (temp_row['actual_price'] - p) / temp_row['actual_price']
            
            d = model.predict(temp_row)[0]
            demands.append(d)
            revenues.append(d * p)
            
        curve_df = pd.DataFrame({'Price': prices, 'Demand': demands, 'Revenue': revenues})
        
        # Plotting
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        sns.lineplot(data=curve_df, x='Price', y='Demand', ax=ax1, color='blue', label='Demand')
        ax1.set_ylabel('Predicted Demand', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        sns.lineplot(data=curve_df, x='Price', y='Revenue', ax=ax2, color='green', label='Revenue')
        ax2.set_ylabel('Projected Revenue (₹)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # Mark selected point
        ax1.axvline(new_price, color='red', linestyle='--', label='Selected Price')
        
        plt.title(f"Price Elasticity for {product_id}")
        st.pyplot(fig)
        
        st.info("The Red dashed line represents your currently selected price slider value.")

if __name__ == "__main__":
    main()
