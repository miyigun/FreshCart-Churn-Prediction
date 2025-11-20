"""
Behavioral Feature Engineering Module
======================================
Features that capture customer behavior patterns.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer:
    """
    Creates customer behavioral features.
    
    Feature Groups:
    - Time-based: Day and hour preferences
    - Reorder behavior: Reordering habits
    - Diversity: Product diversity
    - Consistency: Behavioral consistency
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_behavioral_features(self,
                                       orders_df: pd.DataFrame,
                                       order_products_df: pd.DataFrame,
                                       products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all behavioral features.
        
        Args:
            orders_df: The orders dataframe.
            order_products_df: The order products dataframe.
            products_df: The products dataframe.
            
        Returns:
            A dataframe with user-level behavioral features.
        """
        logger.info("🧠 Creating behavioral features...")
        
        # Time-based features
        time_features = self.create_time_features(orders_df)
        
        # Reorder features
        reorder_features = self.create_reorder_features(orders_df, order_products_df)
        
        # Product diversity features
        diversity_features = self.create_diversity_features(
            orders_df, order_products_df, products_df
        )
        
        # Merge all
        behavioral_features = time_features\
            .merge(reorder_features, on='user_id', how='outer')\
            .merge(diversity_features, on='user_id', how='outer')
        
        # Fill NaN
        behavioral_features = behavioral_features.fillna(0)
        
        self.feature_names = [col for col in behavioral_features.columns if col != 'user_id']
        
        logger.info(f"✅ Created {len(self.feature_names)} behavioral features")
        
        return behavioral_features
    
    def create_time_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Time-based behavioral features.
        
        Features:
        - avg_order_hour: Average order hour.
        - avg_order_dow: Average order day of the week.
        - weekend_order_ratio: Ratio of orders placed on weekends.
        - night_order_ratio: Ratio of orders placed at night (8 PM - 6 AM).
        - morning_order_ratio: Ratio of orders placed in the morning (6 AM - 12 PM).
        - preferred_dow: Most preferred day of the week.
        - preferred_hour: Most preferred hour of the day.
        """
        logger.info("   Creating time-based features...")
        
        # Basic stats
        time_stats = orders_df.groupby('user_id').agg({
            'order_hour_of_day': ['mean', 'std', lambda x: x.mode()[0] if len(x.mode()) > 0 else 0],
            'order_dow': ['mean', 'std', lambda x: x.mode()[0] if len(x.mode()) > 0 else 0]
        }).reset_index()
        
        time_stats.columns = [
            'user_id',
            'avg_order_hour',
            'std_order_hour',
            'preferred_hour',
            'avg_order_dow',
            'std_order_dow',
            'preferred_dow'
        ]
        
        # Weekend orders (dow 5, 6 = Saturday, Sunday)
        weekend_orders = orders_df.groupby('user_id').apply(
            lambda x: (x['order_dow'] >= 5).sum() / len(x)
        ).reset_index()
        weekend_orders.columns = ['user_id', 'weekend_order_ratio']
        
        # Night orders (20-6 hours)
        night_orders = orders_df.groupby('user_id').apply(
            lambda x: ((x['order_hour_of_day'] >= 20) | (x['order_hour_of_day'] < 6)).sum() / len(x)
        ).reset_index()
        night_orders.columns = ['user_id', 'night_order_ratio']
        
        # Morning orders (6-12 hours)
        morning_orders = orders_df.groupby('user_id').apply(
            lambda x: ((x['order_hour_of_day'] >= 6) & (x['order_hour_of_day'] < 12)).sum() / len(x)
        ).reset_index()
        morning_orders.columns = ['user_id', 'morning_order_ratio']
        
        # Afternoon orders (12-18 hours)
        afternoon_orders = orders_df.groupby('user_id').apply(
            lambda x: ((x['order_hour_of_day'] >= 12) & (x['order_hour_of_day'] < 18)).sum() / len(x)
        ).reset_index()
        afternoon_orders.columns = ['user_id', 'afternoon_order_ratio']
        
        # Merge all time features
        time_features = time_stats\
            .merge(weekend_orders, on='user_id')\
            .merge(night_orders, on='user_id')\
            .merge(morning_orders, on='user_id')\
            .merge(afternoon_orders, on='user_id')
        
        return time_features
    
    def create_reorder_features(self,
                               orders_df: pd.DataFrame,
                               order_products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reorder behavior features.
        
        Features:
        - overall_reorder_rate: Overall reorder rate.
        - avg_reorder_rate_per_order: Average reorder rate per order.
        - reorder_consistency: Consistency of reordering.
        - favorite_products_count: Number of favorite products (ordered 5+ times).
        """
        logger.info("   Creating reorder behavior features...")
        
        # Merge to get user_id
        order_products_with_user = order_products_df.merge(
            orders_df[['order_id', 'user_id']], 
            on='order_id'
        )
        
        # Overall reorder rate per user
        reorder_stats = order_products_with_user.groupby('user_id').agg({
            'reordered': ['mean', 'sum', 'std']
        }).reset_index()
        
        reorder_stats.columns = [
            'user_id',
            'overall_reorder_rate',
            'total_reordered_items',
            'reorder_rate_std'
        ]
        
        # Reorder rate per order (some users consistently reorder, others don't)
        reorder_per_order = order_products_with_user.groupby(['user_id', 'order_id'])['reordered'].mean().reset_index()
        reorder_consistency = reorder_per_order.groupby('user_id')['reordered'].agg(['mean', 'std']).reset_index()
        reorder_consistency.columns = ['user_id', 'avg_reorder_rate_per_order', 'reorder_consistency_std']
        
        # Favorite products (ordered 5+ times)
        product_order_counts = order_products_with_user.groupby(['user_id', 'product_id']).size().reset_index()
        product_order_counts.columns = ['user_id', 'product_id', 'times_ordered']
        
        favorite_products = product_order_counts[product_order_counts['times_ordered'] >= 5]\
            .groupby('user_id').size().reset_index()
        favorite_products.columns = ['user_id', 'favorite_products_count']
        
        # Merge
        reorder_features = reorder_stats\
            .merge(reorder_consistency, on='user_id')\
            .merge(favorite_products, on='user_id', how='left')
        
        reorder_features['favorite_products_count'] = reorder_features['favorite_products_count'].fillna(0)
        reorder_features['reorder_rate_std'] = reorder_features['reorder_rate_std'].fillna(0)
        reorder_features['reorder_consistency_std'] = reorder_features['reorder_consistency_std'].fillna(0)
        
        return reorder_features
    
    def create_diversity_features(self,
                                  orders_df: pd.DataFrame,
                                  order_products_df: pd.DataFrame,
                                  products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Product diversity features.
        
        Features:
        - unique_products: Number of unique products.
        - unique_aisles: Number of unique aisles.
        - unique_departments: Number of unique departments.
        - product_diversity_score: Product diversity score.
        - avg_products_per_order: Average products per order.
        - exploration_rate: Rate of trying new products.
        """
        logger.info("   Creating diversity features...")
        
        # Merge to get aisle and department info
        order_products_full = order_products_df\
            .merge(orders_df[['order_id', 'user_id']], on='order_id')\
            .merge(products_df[['product_id', 'aisle_id', 'department_id']], on='product_id')
        
        # Unique counts
        diversity_stats = order_products_full.groupby('user_id').agg({
            'product_id': 'nunique',
            'aisle_id': 'nunique',
            'department_id': 'nunique',
            'order_id': 'nunique'
        }).reset_index()
        
        diversity_stats.columns = [
            'user_id',
            'unique_products',
            'unique_aisles',
            'unique_departments',
            'total_orders'
        ]
        
        # Products per order
        diversity_stats['avg_products_per_order'] = (
            order_products_full.groupby('user_id')['product_id'].count().values / 
            diversity_stats['total_orders']
        )
        
        # Product diversity score (normalized)
        diversity_stats['product_diversity_score'] = (
            diversity_stats['unique_products'] / 
            (diversity_stats['total_orders'] * diversity_stats['avg_products_per_order'] + 1)
        )
        
        # Exploration rate (trying new products over time)
        # Calculate: products in later orders that weren't in the first half
        exploration_rates = []
        
        for user_id in diversity_stats['user_id']:
            user_orders = order_products_full[order_products_full['user_id'] == user_id]\
                .merge(orders_df[['order_id', 'order_number']], on='order_id')
            
            if len(user_orders) > 0:
                mid_point = user_orders['order_number'].median()
                
                early_products = set(user_orders[user_orders['order_number'] <= mid_point]['product_id'])
                late_products = set(user_orders[user_orders['order_number'] > mid_point]['product_id'])
                
                if len(late_products) > 0:
                    exploration_rate = len(late_products - early_products) / len(late_products)
                else:
                    exploration_rate = 0
            else:
                exploration_rate = 0
            
            exploration_rates.append(exploration_rate)
        
        diversity_stats['exploration_rate'] = exploration_rates
        
        # Drop temporary column
        diversity_stats = diversity_stats.drop('total_orders', axis=1)
        
        return diversity_stats
    
    def get_feature_names(self) -> List[str]:
        """Return a list of the feature names."""
        return self.feature_names


def create_behavioral_features_pipeline(orders_df: pd.DataFrame,
                                        order_products_df: pd.DataFrame,
                                        products_df: pd.DataFrame) -> pd.DataFrame:
    """
    A quick pipeline to create all behavioral features.
    
    Args:
        orders_df: The orders dataframe.
        order_products_df: The order products dataframe.
        products_df: The products dataframe.
        
    Returns:
        A dataframe with user-level behavioral features.
    """
    engineer = BehavioralFeatureEngineer()
    behavioral_features = engineer.create_all_behavioral_features(
        orders_df, order_products_df, products_df
    )
    
    return behavioral_features


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
    products_df = data['products']
    
    # Create behavioral features
    behavioral_features = create_behavioral_features_pipeline(
        orders_df, order_products, products_df
    )
    
    print("\n🧠 Behavioral Features Sample:")
    print(behavioral_features.head(10))
    
    print("\n📈 Behavioral Features Statistics:")
    print(behavioral_features.describe())
    
    print("\n✅ Behavioral features created successfully!")