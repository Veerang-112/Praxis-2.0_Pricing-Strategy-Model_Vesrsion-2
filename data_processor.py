import pandas as pd
import numpy as np
import re
# Removed TextBlob dependency to avoid installation issues
# from textblob import TextBlob 

class SimpleSentiment:
    def __init__(self):
        self.positive = {'good', 'great', 'excellent', 'amazing', 'best', 'nice', 'awesome', 'love', 'perfect', 'satisfied', 'value'}
        self.negative = {'bad', 'worst', 'poor', 'terrible', 'waste', 'useless', 'broken', 'slow', 'issue', 'problem', 'return'}

    def get_score(self, text):
        if not isinstance(text, str):
            return 0.0
        words = re.findall(r'\w+', text.lower())
        if not words:
            return 0.0
        
        score = 0
        for word in words:
            if word in self.positive:
                score += 1
            elif word in self.negative:
                score -= 1
        
        # Normalize between -1 and 1 roughly
        return max(min(score / 5.0, 1.0), -1.0)

class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """Loads the dataset from the CSV file."""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def clean_currency(self, value):
        """Cleans currency strings (e.g., '₹399', '₹1,099') to float."""
        if isinstance(value, str):
            # Remove '₹', ',', and any other non-numeric chars except '.'
            cleaned = re.sub(r'[^\d.]', '', value)
            if cleaned:
                return float(cleaned)
        return np.nan

    def clean_percentage(self, value):
        """Cleans percentage strings (e.g., '64%') to float."""
        if isinstance(value, str):
            cleaned = re.sub(r'[^\d.]', '', value)
            if cleaned:
                return float(cleaned) / 100
        return np.nan

    def clean_number(self, value):
        """Cleans number strings with commas (e.g., '24,269') to float."""
        if isinstance(value, str):
            cleaned = re.sub(r'[^\d]', '', value)
            if cleaned:
                return float(cleaned)
        elif isinstance(value, (int, float)):
             return float(value)
        return np.nan

    def get_sentiment(self, text):
        """Calculates sentiment polarity of the text using simple dictionary."""
        analyzer = SimpleSentiment() # Inefficient to init here every time but safe
        return analyzer.get_score(text)

    def extract_categories(self, category_str):
        """Extracts main and sub categories from the category string."""
        if isinstance(category_str, str):
            parts = category_str.split('|')
            main = parts[0] if len(parts) > 0 else 'Other'
            sub = parts[1] if len(parts) > 1 else 'Other'
            return main, sub
        return 'Other', 'Other'

    def process(self):
        """Runs the full data processing pipeline."""
        if self.df is None:
            self.load_data()

        print("Starting data cleaning...")
        
        # Clean Prices
        self.df['discounted_price'] = self.df['discounted_price'].apply(self.clean_currency)
        self.df['actual_price'] = self.df['actual_price'].apply(self.clean_currency)
        self.df['discount_percentage'] = self.df['discount_percentage'].apply(self.clean_percentage)
        
        # Clean Rating Count
        self.df['rating_count'] = self.df['rating_count'].apply(self.clean_number)
        
        # Clean Rating (handle inconsistent types if any)
        self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce')

        # Handle Missing Values
        # Drop rows where critical target/feature info is missing
        self.df.dropna(subset=['discounted_price', 'actual_price', 'rating_count', 'rating'], inplace=True)

        print("Starting feature engineering...")
        
        # Feature Engineering: Categories
        self.df[['category_main', 'category_sub']] = self.df['category'].apply(
            lambda x: pd.Series(self.extract_categories(x))
        )

        # Feature Engineering: Price Gap
        self.df['price_gap'] = self.df['actual_price'] - self.df['discounted_price']

        # Feature Engineering: Sentiment
        # Use review_content if available, else review_title. 
        # (Assuming 'review_content' contains the detailed review)
        self.df['review_text'] = self.df['review_content'].fillna('') + " " + self.df['review_title'].fillna('')
        self.df['sentiment_score'] = self.df['review_text'].apply(self.get_sentiment)

        # Drop intermediate text columns to save memory if needed (optional, keeping for now for verification)
        # self.df.drop(columns=['review_content', 'review_title', 'review_text'], inplace=True)

        print(f"Data processing complete. Final Shape: {self.df.shape}")
        return self.df

if __name__ == "__main__":
    # Test run
    processor = DataProcessor(r"c:\Users\DELL\OneDrive\Desktop\anti\pricing_dataset.csv")
    cleaned_df = processor.process()
    print(cleaned_df.head())
