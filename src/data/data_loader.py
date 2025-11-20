"""
Data Loading Module
===================
Functions to load and merge the Instacart dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstacartDataLoader:
    """Class to load Instacart data."""
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Directory where the raw data files are located.
        """
        self.data_dir = Path(data_dir)
        self.data = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all Instacart CSV files.
        
        Returns:
            A dictionary containing all dataframes.
        """
        logger.info("📦 Loading Instacart datasets...")
        
        files = {
            'orders': 'orders.csv',
            'order_products_prior': 'order_products__prior.csv',
            'order_products_train': 'order_products__train.csv',
            'products': 'products.csv',
            'aisles': 'aisles.csv',
            'departments': 'departments.csv'
        }
        
        for key, filename in files.items():
            filepath = self.data_dir / filename
            logger.info(f"   Loading {filename}...")
            
            try:
                self.data[key] = pd.read_csv(filepath)
                logger.info(f"   ✅ Loaded {key}: {self.data[key].shape}")
            except FileNotFoundError:
                logger.error(f"   ❌ File not found: {filepath}")
                raise
            except Exception as e:
                logger.error(f"   ❌ Error loading {filename}: {str(e)}")
                raise
        
        logger.info(f"✅ All datasets loaded successfully!\n")
        self._print_data_summary()
        
        return self.data
    
    def _print_data_summary(self):
        """Prints a summary of the loaded data."""
        logger.info("=" * 80)
        logger.info("DATA SUMMARY")
        logger.info("=" * 80)
        
        for name, df in self.data.items():
            logger.info(f"{name:25s}: {df.shape[0]:>10,} rows x {df.shape[1]:>3} columns")
            logger.info(f"{'':25s}  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        logger.info("=" * 80 + "\n")
    
    def merge_order_products(self) -> pd.DataFrame:
        """
        Merges the prior and train order_products dataframes.
        
        Returns:
            The merged order_products dataframe.
        """
        logger.info("🔗 Merging order_products datasets...")
        
        order_products = pd.concat([
            self.data['order_products_prior'],
            self.data['order_products_train']
        ], ignore_index=True)
        
        logger.info(f"✅ Merged order_products: {order_products.shape}")
        
        return order_products
    
    def create_master_dataset(self) -> pd.DataFrame:
        """
        Creates a master dataset by merging all tables.
        
        Returns:
            A master dataframe with all information.
        """
        logger.info("🏗️  Creating master dataset...")
        
        # Merge order_products
        order_products = self.merge_order_products()
        
        # Add product information
        logger.info("   Merging with products...")
        df = order_products.merge(
            self.data['products'],
            on='product_id',
            how='left'
        )
        
        # Add aisle information
        logger.info("   Merging with aisles...")
        df = df.merge(
            self.data['aisles'],
            on='aisle_id',
            how='left'
        )
        
        # Add department information
        logger.info("   Merging with departments...")
        df = df.merge(
            self.data['departments'],
            on='department_id',
            how='left'
        )
        
        # Add order information
        logger.info("   Merging with orders...")
        df = df.merge(
            self.data['orders'],
            on='order_id',
            how='left'
        )
        
        logger.info(f"✅ Master dataset created: {df.shape}")
        logger.info(f"   Columns: {list(df.columns)}\n")
        
        return df
    
    def get_user_order_history(self, user_id: int) -> pd.DataFrame:
        """
        Retrieves the order history for a specific user.
        
        Args:
            user_id: The User ID.
            
        Returns:
            The user's order history as a dataframe.
        """
        master_df = self.create_master_dataset()
        user_history = master_df[master_df['user_id'] == user_id].copy()
        
        logger.info(f"📊 User {user_id} history: {user_history.shape[0]} orders")
        
        return user_history
    
    def get_data_info(self) -> Dict:
        """
        Provides detailed information about the data.
        
        Returns:
            A dictionary with data statistics.
        """
        if not self.data:
            logger.warning("⚠️  No data loaded yet!")
            return {}
        
        info = {
            'total_orders': len(self.data['orders']),
            'total_users': self.data['orders']['user_id'].nunique(),
            'total_products': len(self.data['products']),
            'total_aisles': len(self.data['aisles']),
            'total_departments': len(self.data['departments']),
            'avg_orders_per_user': len(self.data['orders']) / self.data['orders']['user_id'].nunique(),
            'date_range': (
                self.data['orders']['order_dow'].min(),
                self.data['orders']['order_dow'].max()
            )
        }
        
        return info
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, 
                           output_dir: Optional[Path] = None):
        """
        Saves the processed data.
        
        Args:
            df: The dataframe to save.
            filename: The output filename.
            output_dir: The output directory (default: data/processed).
        """
        if output_dir is None:
            output_dir = self.data_dir.parent / "processed"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        
        logger.info(f"💾 Saving to {filepath}...")
        
        if filepath.suffix == '.parquet':
            df.to_parquet(filepath, index=False)
        elif filepath.suffix == '.csv':
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        logger.info(f"✅ Saved successfully!")


def load_instacart_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    A quick loader function.
    
    Args:
        data_dir: The directory containing Instacart CSV files.
        
    Returns:
        A dictionary of dataframes.
    """
    loader = InstacartDataLoader(data_dir)
    return loader.load_all_data()


def create_sample_data(data: Dict[str, pd.DataFrame], 
                      sample_size: int = 10000) -> Dict[str, pd.DataFrame]:
    """
    Creates sample data for quick testing.
    
    Args:
        data: The original data dictionary.
        sample_size: The number of users to sample.
        
    Returns:
        A sampled data dictionary.
    """
    logger.info(f"🎲 Creating sample dataset with {sample_size} users...")
    
    # Sample users
    sample_users = data['orders']['user_id'].drop_duplicates().sample(
        n=min(sample_size, data['orders']['user_id'].nunique()),
        random_state=42
    )
    
    # Filter orders
    sample_orders = data['orders'][
        data['orders']['user_id'].isin(sample_users)
    ].copy()
    
    sample_order_ids = sample_orders['order_id'].unique()
    
    # Filter order_products
    sample_op_prior = data['order_products_prior'][
        data['order_products_prior']['order_id'].isin(sample_order_ids)
    ].copy()
    
    sample_op_train = data['order_products_train'][
        data['order_products_train']['order_id'].isin(sample_order_ids)
    ].copy()
    
    sampled_data = {
        'orders': sample_orders,
        'order_products_prior': sample_op_prior,
        'order_products_train': sample_op_train,
        'products': data['products'].copy(),
        'aisles': data['aisles'].copy(),
        'departments': data['departments'].copy()
    }
    
    logger.info(f"✅ Sample dataset created:")
    logger.info(f"   Users: {len(sample_users):,}")
    logger.info(f"   Orders: {len(sample_orders):,}")
    logger.info(f"   Products in orders: {len(sample_op_prior) + len(sample_op_train):,}\n")
    
    return sampled_data


# Example usage
if __name__ == "__main__":
    from pathlib import Path
    
    # Example: Load data
    data_dir = Path("../../data/raw")
    
    if data_dir.exists():
        loader = InstacartDataLoader(data_dir)
        data = loader.load_all_data()
        
        # Create master dataset
        master_df = loader.create_master_dataset()
        
        # Save processed data
        loader.save_processed_data(master_df, "master_dataset.parquet")
        
        # Print info
        info = loader.get_data_info()
        print("\n📊 Dataset Statistics:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    else:
        print(f"❌ Data directory not found: {data_dir}")
        print("   Please download Instacart data first!")