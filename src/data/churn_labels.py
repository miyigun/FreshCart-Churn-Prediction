"""
Churn Label Creation Module
============================
Functions to determine the churn status of customers.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging
from datetime import timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnLabelCreator:
    """
    Class that creates churn labels.
    
    Churn Definition:
    - Customers who have not ordered for 30+ days since their last order are considered churned.
    - Must have a history of at least 3 orders (new customers are excluded from the analysis).
    - We use a 90-day observation window.
    """
    
    def __init__(self, 
                 churn_threshold_days: int = 30,
                 min_orders: int = 3,
                 observation_window_days: int = 90):
        """
        Args:
            churn_threshold_days: Day threshold for churn.
            min_orders: Minimum number of orders.
            observation_window_days: Observation window in days.
        """
        self.churn_threshold = churn_threshold_days
        self.min_orders = min_orders
        self.observation_window = observation_window_days
        
        logger.info(f"🎯 Churn Definition:")
        logger.info(f"   Threshold: {self.churn_threshold} days")
        logger.info(f"   Min Orders: {self.min_orders}")
        logger.info(f"   Observation Window: {self.observation_window} days")
    
    def create_user_order_summary(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create an order summary for each user.
        
        Args:
            orders_df: The orders dataframe.
            
        Returns:
            A user-level aggregated dataframe.
        """
        logger.info("📊 Creating user order summary...")
        
        # Basic metrics for each user
        user_summary = orders_df.groupby('user_id').agg({
            'order_id': 'count',
            'order_number': 'max',
            'days_since_prior_order': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        # Flatten column names
        user_summary.columns = [
            'user_id', 'total_orders', 'max_order_number',
            'avg_days_between_orders', 'std_days_between_orders',
            'min_days_between_orders', 'max_days_between_orders'
        ]
        
        # First and last order for each user (based on order_number)
        first_orders = orders_df.groupby('user_id')['order_number'].min().reset_index()
        first_orders.columns = ['user_id', 'first_order_number']
        
        last_orders = orders_df.groupby('user_id')['order_number'].max().reset_index()
        last_orders.columns = ['user_id', 'last_order_number']
        
        user_summary = user_summary.merge(first_orders, on='user_id')
        user_summary = user_summary.merge(last_orders, on='user_id')
        
        logger.info(f"✅ User summary created: {user_summary.shape}")
        
        return user_summary
    
    def calculate_recency(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate recency (days since last order) for each user.
        
        Args:
            orders_df: The orders dataframe.
            
        Returns:
            A dataframe with user_id and recency.
        """
        logger.info("📅 Calculating recency for each user...")
        
        # Find the maximum order_number for each user (the most recent order)
        max_order_per_user = orders_df.groupby('user_id')['order_number'].max().reset_index()
        #max_order_per_user.columns = ['user_id', 'max_order_number']
        
        # Global maximum order_number in the dataset (reference point)
        # This is considered the "current" point in time.
        global_max_order = orders_df['order_number'].max()
        
        # Find the last order for each user
        last_orders = orders_df.merge(
            max_order_per_user,
            left_on=['user_id', 'order_number'],
            right_on=['user_id', 'order_number']
        )

        # Simplified approach: use the difference between last order_number and global max
        recency_df = orders_df.groupby('user_id').agg({
            'order_number': 'max'
        }).reset_index()
        
        recency_df.columns = ['user_id', 'last_order_number']
        
        # Normalize against the global last order
        # If a user's last order is far from the global max, they have likely churned.
        recency_df['orders_behind'] = global_max_order - recency_df['last_order_number']
        
        # Estimated recency (very simplified)
        # Should be more sophisticated in a real implementation.
        recency_df['estimated_recency_days'] = recency_df['orders_behind'] * 7  # Assuming an average of 7 days/order
        
        logger.info(f"✅ Recency calculated for {len(recency_df)} users")
        
        return recency_df[['user_id', 'estimated_recency_days', 'orders_behind']]
    
    def create_churn_labels(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates the churn labels.
        
        Args:
            orders_df: The orders dataframe.
            
        Returns:
            A dataframe with user_id and churn label (0/1).
        """
        logger.info("🏷️  Creating churn labels...")
        
        # User summary
        user_summary = self.create_user_order_summary(orders_df)
        
        # Calculate recency
        recency_df = self.calculate_recency(orders_df)
        
        # Merge
        user_data = user_summary.merge(recency_df, on='user_id')
        
        # Create churn label
        # Criterion 1: Must have at least min_orders
        # Criterion 2: estimated_recency_days > churn_threshold
        
        user_data['is_churn'] = (
            (user_data['total_orders'] >= self.min_orders) &
            (user_data['estimated_recency_days'] > self.churn_threshold)
        ).astype(int)
        
        # Statistics
        total_users = len(user_data)
        eligible_users = (user_data['total_orders'] >= self.min_orders).sum()
        churned_users = user_data['is_churn'].sum()
        churn_rate = churned_users / eligible_users * 100 if eligible_users > 0 else 0
        
        logger.info(f"\n{'='*80}")
        logger.info(f"CHURN LABEL STATISTICS")
        logger.info(f"{'='*80}")
        logger.info(f"Total Users:           {total_users:>10,}")
        logger.info(f"Eligible Users:        {eligible_users:>10,} (min {self.min_orders} orders)")
        logger.info(f"Churned Users:         {churned_users:>10,}")
        logger.info(f"Active Users:          {eligible_users - churned_users:>10,}")
        logger.info(f"Churn Rate:            {churn_rate:>10.2f}%")
        logger.info(f"{'='*80}\n")
        
        return user_data[['user_id', 'is_churn', 'estimated_recency_days', 
                          'total_orders', 'orders_behind']]
    
    def split_train_test_temporal(self, 
                                  orders_df: pd.DataFrame,
                                  test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Perform a time-based train-test split.
        
        Separates the last X% of orders as the test set to prevent data leakage.
        
        Args:
            orders_df: The orders dataframe.
            test_size: The proportion of the dataset to include in the test split.
            
        Returns:
            A tuple of (train_orders, test_orders).
        """
        logger.info(f"⏱️  Creating temporal train-test split ({test_size:.0%} test)...")
        
        # Sort by order_number
        orders_sorted = orders_df.sort_values('order_number')
        
        # Determine the split point
        split_idx = int(len(orders_sorted) * (1 - test_size))
        split_order_number = orders_sorted.iloc[split_idx]['order_number']
        
        # Split
        train_orders = orders_df[orders_df['order_number'] <= split_order_number].copy()
        test_orders = orders_df[orders_df['order_number'] > split_order_number].copy()
        
        logger.info(f"✅ Train orders: {len(train_orders):,} ({len(train_orders)/len(orders_df):.1%})")
        logger.info(f"✅ Test orders:  {len(test_orders):,} ({len(test_orders)/len(orders_df):.1%})")
        logger.info(f"   Split at order_number: {split_order_number}")
        
        return train_orders, test_orders
    
    def create_labels_for_split(self, 
                                train_orders: pd.DataFrame,
                                test_orders: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create labels for train and test sets separately.
        
        Args:
            train_orders: The training orders dataframe.
            test_orders: The test orders dataframe.
            
        Returns:
            A tuple of (train_labels, test_labels).
        """
        logger.info("🔀 Creating labels for train and test splits...")
        
        # Train labels
        train_labels = self.create_churn_labels(train_orders)
        train_labels['split'] = 'train'
        
        # Test labels
        test_labels = self.create_churn_labels(test_orders)
        test_labels['split'] = 'test'
        
        return train_labels, test_labels


def analyze_churn_distribution(labels_df: pd.DataFrame) -> Dict:
    """
    Analyzes the churn distribution.
    
    Args:
        labels_df: The churn labels dataframe.
        
    Returns:
        A dictionary with statistics.
    """
    stats = {
        'total_users': len(labels_df),
        'churned_users': labels_df['is_churn'].sum(),
        'active_users': (labels_df['is_churn'] == 0).sum(),
        'churn_rate': labels_df['is_churn'].mean(),
        'avg_recency_churned': labels_df[labels_df['is_churn'] == 1]['estimated_recency_days'].mean(),
        'avg_recency_active': labels_df[labels_df['is_churn'] == 0]['estimated_recency_days'].mean(),
        'avg_orders_churned': labels_df[labels_df['is_churn'] == 1]['total_orders'].mean(),
        'avg_orders_active': labels_df[labels_df['is_churn'] == 0]['total_orders'].mean()
    }
    
    print("\n" + "="*80)
    print("CHURN DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"Total Users:                    {stats['total_users']:>10,}")
    print(f"Churned Users:                  {stats['churned_users']:>10,} ({stats['churn_rate']:.2%})")
    print(f"Active Users:                   {stats['active_users']:>10,} ({1-stats['churn_rate']:.2%})")
    print(f"\nAvg Recency (Churned):          {stats['avg_recency_churned']:>10.1f} days")
    print(f"Avg Recency (Active):           {stats['avg_recency_active']:>10.1f} days")
    print(f"\nAvg Orders (Churned):           {stats['avg_orders_churned']:>10.1f}")
    print(f"Avg Orders (Active):            {stats['avg_orders_active']:>10.1f}")
    print("="*80)
    
    return stats


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
    
    # Create churn labels
    creator = ChurnLabelCreator(
        churn_threshold_days=30,
        min_orders=3,
        observation_window_days=90
    )
    
    # Full dataset labels
    labels = creator.create_churn_labels(orders_df)
    
    # Analyze
    stats = analyze_churn_distribution(labels)
    
    # Train-test split with labels
    train_orders, test_orders = creator.split_train_test_temporal(orders_df, test_size=0.2)
    train_labels, test_labels = creator.create_labels_for_split(train_orders, test_orders)
    
    print("\n✅ Churn labels created successfully!")