"""
RFM Feature Engineering Module
===============================
Creates Recency, Frequency, and Monetary features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMFeatureEngineer:
    """
    Creates RFM (Recency, Frequency, Monetary) features.
    
    Features:
    - Recency: Days since the last order.
    - Frequency: Order frequency.
    - Monetary: Monetary value (using basket size as a proxy).
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_rfm_features(self, 
                                orders_df: pd.DataFrame,
                                order_products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all RFM features.
        
        Args:
            orders_df: The orders dataframe.
            order_products_df: The order products dataframe.
            
        Returns:
            A dataframe with user-level RFM features.
        """
        logger.info("🔧 Creating RFM features...")
        
        # Recency features
        recency_features = self.create_recency_features(orders_df)
        
        # Frequency features
        frequency_features = self.create_frequency_features(orders_df)
        
        # Monetary features (using basket size as a proxy)
        monetary_features = self.create_monetary_features(orders_df, order_products_df)
        
        # Merge all
        rfm_features = recency_features\
            .merge(frequency_features, on='user_id', how='outer')\
            .merge(monetary_features, on='user_id', how='outer')
        
        # Fill NaN with 0
        rfm_features = rfm_features.fillna(0)
        
        self.feature_names = [col for col in rfm_features.columns if col != 'user_id']
        
        logger.info(f"✅ Created {len(self.feature_names)} RFM features")
        logger.info(f"   Features: {self.feature_names}")
        
        return rfm_features
    
    def create_recency_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Recency features.
        
        Features:
        - days_since_last_order: Days since the last order.
        - days_since_first_order: Days since the first order.
        - customer_age_days: Customer age in days.
        """
        logger.info("   Creating recency features...")
        
        user_recency = orders_df.groupby('user_id').agg({
            'order_number': ['min', 'max'],
            'days_since_prior_order': ['mean', 'sum']
        }).reset_index()
        
        user_recency.columns = [
            'user_id', 
            'first_order_number', 
            'last_order_number',
            'avg_days_between_orders',
            'total_days_since_prior'
        ]
        
        # Global max order number (reference point - "now")
        global_max = orders_df['order_number'].max()
        
        # Recency calculations
        user_recency['orders_since_last'] = global_max - user_recency['last_order_number']
        user_recency['days_since_last_order'] = user_recency['orders_since_last'] * 7  # Estimate
        
        user_recency['total_order_span'] = user_recency['last_order_number'] - user_recency['first_order_number']
        user_recency['customer_age_days'] = user_recency['total_order_span'] * 7  # Estimate
        
        # Days since first order
        user_recency['days_since_first_order'] = user_recency['customer_age_days'] + user_recency['days_since_last_order']
        
        # Select final features
        recency_cols = [
            'user_id',
            'days_since_last_order',
            'days_since_first_order', 
            'customer_age_days',
            'avg_days_between_orders'
        ]
        
        return user_recency[recency_cols]
    
    def create_frequency_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Frequency features.
        
        Features:
        - total_orders: Total number of orders.
        - orders_per_day: Average orders per day.
        - order_frequency: Order frequency score.
        - order_regularity: Order regularity (low std means more regular).
        """
        logger.info("   Creating frequency features...")
        
        user_frequency = orders_df.groupby('user_id').agg({
            'order_id': 'count',
            'order_number': ['min', 'max'],
            'days_since_prior_order': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        user_frequency.columns = [
            'user_id',
            'total_orders',
            'first_order_number',
            'last_order_number',
            'avg_days_between_orders',
            'std_days_between_orders',
            'min_days_between_orders',
            'max_days_between_orders'
        ]
        
        # Derived features
        user_frequency['order_span'] = user_frequency['last_order_number'] - user_frequency['first_order_number']
        user_frequency['estimated_customer_days'] = user_frequency['order_span'] * 7
        
        # Orders per day (frequency rate)
        user_frequency['orders_per_day'] = user_frequency['total_orders'] / (user_frequency['estimated_customer_days'] + 1)
        
        # Order regularity (coefficient of variation)
        user_frequency['order_regularity'] = (
            user_frequency['std_days_between_orders'] / 
            (user_frequency['avg_days_between_orders'] + 1)
        )
        
        # Fill NaN in std (happens when there are only 1-2 orders)
        user_frequency['std_days_between_orders'] = user_frequency['std_days_between_orders'].fillna(0)
        user_frequency['order_regularity'] = user_frequency['order_regularity'].fillna(0)
        
        # Select final features
        frequency_cols = [
            'user_id',
            'total_orders',
            'orders_per_day',
            'order_regularity',
            'std_days_between_orders'
        ]
        
        return user_frequency[frequency_cols]
    
    def create_monetary_features(self, 
                                 orders_df: pd.DataFrame,
                                 order_products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Monetary features.
        
        Note: There is no price information, so we use basket size as a proxy.
        
        Features:
        - avg_basket_size: Average basket size (number of products).
        - total_products_ordered: Total number of products ordered.
        - avg_unique_products: Average unique products per order.
        - basket_size_std: Variability of basket size.
        """
        logger.info("   Creating monetary features (using basket size as a proxy)...")
        
        # Calculate basket size per order
        basket_sizes = order_products_df.groupby('order_id').agg({
            'product_id': ['count', 'nunique']
        }).reset_index()
        
        basket_sizes.columns = ['order_id', 'basket_size', 'unique_products_in_order']
        
        # Merge with orders to get user_id
        baskets_with_user = orders_df[['order_id', 'user_id']].merge(
            basket_sizes, on='order_id', how='left'
        )
        
        # User-level aggregation
        user_monetary = baskets_with_user.groupby('user_id').agg({
            'basket_size': ['mean', 'sum', 'std', 'min', 'max'],
            'unique_products_in_order': ['mean', 'sum']
        }).reset_index()
        
        user_monetary.columns = [
            'user_id',
            'avg_basket_size',
            'total_items_ordered',
            'basket_size_std',
            'min_basket_size',
            'max_basket_size',
            'avg_unique_products_per_order',
            'total_unique_products_ordered'
        ]
        
        # Fill NaN
        user_monetary['basket_size_std'] = user_monetary['basket_size_std'].fillna(0)
        
        # Basket size consistency (lower is more consistent)
        user_monetary['basket_size_cv'] = (
            user_monetary['basket_size_std'] / 
            (user_monetary['avg_basket_size'] + 1)
        )
        
        # Select final features
        monetary_cols = [
            'user_id',
            'avg_basket_size',
            'total_items_ordered',
            'basket_size_std',
            'basket_size_cv',
            'avg_unique_products_per_order',
            'total_unique_products_ordered'
        ]
        
        return user_monetary[monetary_cols]
    
    def create_rfm_score(self, rfm_features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the RFM score (on a scale of 1-5).
        
        RFM Score = Recency Score + Frequency Score + Monetary Score
        A high score indicates a valuable customer.
        
        Args:
            rfm_features: The RFM features dataframe.
            
        Returns:
            A dataframe with RFM scores.
        """
        logger.info("📊 Calculating RFM scores...")
        
        rfm_scored = rfm_features.copy()
        
        # Recency score (lower is better, so we invert the labels)
        rfm_scored['recency_score'] = pd.qcut(
            rfm_scored['days_since_last_order'], 
            q=5, 
            labels=[5, 4, 3, 2, 1],
            duplicates='drop'
        )
        
        # Frequency score (higher is better)
        rfm_scored['frequency_score'] = pd.qcut(
            rfm_scored['total_orders'], 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        )
        
        # Monetary score (higher is better)
        rfm_scored['monetary_score'] = pd.qcut(
            rfm_scored['avg_basket_size'], 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        )
        
        # Convert to int
        rfm_scored['recency_score'] = rfm_scored['recency_score'].astype(int)
        rfm_scored['frequency_score'] = rfm_scored['frequency_score'].astype(int)
        rfm_scored['monetary_score'] = rfm_scored['monetary_score'].astype(int)
        
        # Overall RFM score
        rfm_scored['rfm_score'] = (
            rfm_scored['recency_score'] + 
            rfm_scored['frequency_score'] + 
            rfm_scored['monetary_score']
        )
        
        # RFM segment (simplified)
        rfm_scored['rfm_segment'] = pd.cut(
            rfm_scored['rfm_score'],
            bins=[0, 6, 9, 12, 15],
            labels=['At Risk', 'Promising', 'Loyal', 'Champions']
        )
        
        logger.info(f"✅ RFM scores calculated")
        logger.info(f"\nRFM Segment Distribution:")
        print(rfm_scored['rfm_segment'].value_counts().sort_index())
        
        return rfm_scored
    
    def get_feature_names(self) -> List[str]:
        """Return a list of the feature names."""
        return self.feature_names


def create_rfm_features_pipeline(orders_df: pd.DataFrame,
                                 order_products_df: pd.DataFrame) -> pd.DataFrame:
    """
    A quick pipeline to create all RFM features.
    
    Args:
        orders_df: The orders dataframe.
        order_products_df: The order products dataframe.
        
    Returns:
        A dataframe with user-level RFM features and scores.
    """
    engineer = RFMFeatureEngineer()
    
    # Create features
    rfm_features = engineer.create_all_rfm_features(orders_df, order_products_df)
    
    # Add RFM scores
    rfm_with_scores = engineer.create_rfm_score(rfm_features)
    
    return rfm_with_scores


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append('../../')
    from src.data.data_loader import InstacartDataLoader
    from src.config import RAW_DATA_DIR
    
    # Load data
    loader = InstacartDataLoader(RAW_DATA_DIR)
    data = loader.load_all_data()
    
    orders_df = data['orders']
    order_products = pd.concat([
        data['order_products_prior'],
        data['order_products_train']
    ])
    
    # Create RFM features
    rfm_features = create_rfm_features_pipeline(orders_df, order_products)
    
    print("\n📊 RFM Features Sample:")
    print(rfm_features.head(10))
    
    print("\n📈 RFM Features Statistics:")
    print(rfm_features.describe())
    
    print("\n✅ RFM features created successfully!")