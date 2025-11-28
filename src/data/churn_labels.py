"""
Churn Label Creation Module (CORRECTED - HOLD OUT STRATEGY)
===========================================================
Uses the 'train' evaluation set to define ground truth churn.
Prevents Data Leakage by separating 'prior' (history) and 'train' (target).
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnLabelCreator:
    """
    Creates churn labels based on the 'train' evaluation set provided by Instacart.
    
    New Strategy (Leakage-Free):
    1. TARGET (Label): The 'train' set rows represent the users' NEXT order.
       - If 'days_since_prior_order' in 'train' set >= churn_threshold -> CHURN (1)
       - If 'days_since_prior_order' in 'train' set < churn_threshold -> ACTIVE (0)
       
    2. FEATURES: Calculated ONLY from 'prior' set rows.
    """
    
    def __init__(self, churn_threshold_days: int = 30):
        """
        Args:
            churn_threshold_days: If days since prior order >= this, user is churned.
                                  Note: Instacart data caps this at 30, so 30 means '30+ days'.
        """
        self.churn_threshold = churn_threshold_days
        logger.info(f"🎯 Churn Definition Strategy: Next Order Prediction")
        logger.info(f"   Threshold: days_since_prior_order >= {self.churn_threshold}")

    def create_churn_labels(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates labels using ONLY the 'train' set rows which represent the 'next' order.
        
        Args:
            orders_df: The full orders dataframe (must contain 'eval_set' column).
            
        Returns:
            A dataframe with ['user_id', 'is_churn', 'days_to_next_order'].
        """
        logger.info("🏷️  Creating churn labels using 'train' set target...")
        
        # 1. Filter only the 'train' set rows. These are our targets.
        # Note: 'test' set rows have no labels (Kaggle submission), so we ignore them here.
        train_targets = orders_df[orders_df['eval_set'] == 'train'].copy()
        
        if train_targets.empty:
            logger.error("❌ No 'train' rows found in orders_df! Ensure data is loaded correctly.")
            raise ValueError("No 'train' evaluation set found.")

        # 2. Define Target Variable
        # Handle NaN (first orders shouldn't be in train set, but good for safety)
        train_targets['days_since_prior_order'] = train_targets['days_since_prior_order'].fillna(0)
        
        # Create label: 1 if churned (>= 30 days), 0 if active (< 30 days)
        train_targets['is_churn'] = (
            train_targets['days_since_prior_order'] >= self.churn_threshold
        ).astype(int)
        
        # 3. Keep relevant columns
        # We keep 'days_since_prior_order' as 'days_to_next_order' for analysis
        labels_df = train_targets[['user_id', 'is_churn', 'days_since_prior_order']].rename(
            columns={'days_since_prior_order': 'days_to_next_order'}
        )
        
        # 4. Statistics
        self._print_stats(labels_df)
        
        return labels_df

    def _print_stats(self, labels_df: pd.DataFrame):
        """Helper to print label distribution statistics."""
        total_users = len(labels_df)
        churn_cnt = labels_df['is_churn'].sum()
        active_cnt = total_users - churn_cnt
        churn_rate = churn_cnt / total_users
        
        logger.info(f"\n{'='*80}")
        logger.info(f"CHURN LABEL STATISTICS (Ground Truth)")
        logger.info(f"{'='*80}")
        logger.info(f"Total Target Users:      {total_users:>10,}")
        logger.info(f"Churned (>=30 days):     {churn_cnt:>10,} ({churn_rate:.2%})")
        logger.info(f"Active (<30 days):       {active_cnt:>10,} ({1-churn_rate:.2%})")
        logger.info(f"{'='*80}\n")

    def split_train_test_stratified(self, 
                                    master_df: pd.DataFrame, 
                                    test_size: float = 0.2,
                                    random_state: int = 42):
        """
        Performs stratified train-test split on the final dataset.
        Since we rely on the dataset's inherent 'train' split for labels,
        we just split users randomly here to validate our model.
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"✂️  Splitting data (Test size: {test_size}, Stratified)...")
        
        X = master_df.drop(['user_id', 'is_churn', 'eval_set', 'days_to_next_order'], axis=1, errors='ignore')
        y = master_df['is_churn']
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# Example usage for testing
if __name__ == "__main__":
    # Mock data creation for testing logic
    # In real usage, load actual data
    print("🧪 Testing ChurnLabelCreator logic...")
    
    mock_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'eval_set': ['train', 'train', 'train', 'prior', 'test'],
        'days_since_prior_order': [30.0, 7.0, 14.0, 5.0, 0.0]
    })
    
    creator = ChurnLabelCreator(churn_threshold_days=30)
    labels = creator.create_churn_labels(mock_data)
    
    print("\nResulting Labels:")
    print(labels)
    
    # Expected: 
    # User 1: Churn (30 >= 30)
    # User 2: Active (7 < 30)
    # User 3: Active (14 < 30)
    # User 4: Ignored (prior)
    # User 5: Ignored (test)