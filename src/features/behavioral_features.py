"""
Davranışsal Özellik Mühendisliği Modülü
======================================
Müşteri davranış kalıplarını yakalayan özellikler.
"""

import pandas as pd
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehavioralFeatureEngineer:
    """
    Müşteri davranışsal özelliklerini oluşturur.
    
    Özellik Grupları:
    - Zaman bazlı: Gün ve saat tercihleri
    - Tekrar sipariş davranışı: Tekrar sipariş alışkanlıkları
    - Çeşitlilik: Ürün çeşitliliği
    - Tutarlılık: Davranışsal tutarlılık
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_behavioral_features(self,
                                       orders_df: pd.DataFrame,
                                       order_products_df: pd.DataFrame,
                                       products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm davranışsal özellikleri oluşturur.
        
        Args:
            orders_df: Siparişler veri çerçevesi.
            order_products_df: Sipariş ürünleri veri çerçevesi.
            products_df: Ürünler veri çerçevesi.
            
        Returns:
            Kullanıcı düzeyinde davranışsal özelliklere sahip bir veri çerçevesi.
        """
        logger.info("Davranışsal özellikler oluşturuluyor...")
        
        # Zaman bazlı özellikler
        time_features = self.create_time_features(orders_df)
        
        # Tekrar sipariş özellikleri
        reorder_features = self.create_reorder_features(orders_df, order_products_df)
        
        # Ürün çeşitliliği özellikleri
        diversity_features = self.create_diversity_features(
            orders_df, order_products_df, products_df
        )
        
        # Hepsini birleştir
        behavioral_features = time_features\
            .merge(reorder_features, on='user_id', how='outer')\
            .merge(diversity_features, on='user_id', how='outer')
        
        # NaN değerlerini doldur
        behavioral_features = behavioral_features.fillna(0)
        
        self.feature_names = [col for col in behavioral_features.columns if col != 'user_id']
        
        logger.info(f"{len(self.feature_names)} adet davranışsal özellik oluşturuldu")
        
        return behavioral_features
    
    def create_time_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Zaman bazlı davranışsal özellikler.
        
        Özellikler:
        - avg_order_hour: Ortalama sipariş saati.
        - avg_order_dow: Ortalama sipariş haftanın günü.
        - weekend_order_ratio: Hafta sonu verilen siparişlerin oranı.
        - night_order_ratio: Gece verilen siparişlerin oranı (20:00 - 06:00).
        - morning_order_ratio: Sabah verilen siparişlerin oranı (06:00 - 12:00).
        - preferred_dow: En çok tercih edilen haftanın günü.
        - preferred_hour: En çok tercih edilen günün saati.
        """
        logger.info("Zaman bazlı özellikler oluşturuluyor...")
        
        # Temel istatistikler
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
        
        # Hafta sonu siparişleri (dow 5, 6 = Cumartesi, Pazar)
        weekend_orders = orders_df.groupby('user_id').apply(
            lambda x: (x['order_dow'] >= 5).sum() / len(x)
        ).reset_index()
        weekend_orders.columns = ['user_id', 'weekend_order_ratio']
        
        # Gece siparişleri (20-06 saatleri)
        night_orders = orders_df.groupby('user_id').apply(
            lambda x: ((x['order_hour_of_day'] >= 20) | (x['order_hour_of_day'] < 6)).sum() / len(x)
        ).reset_index()
        night_orders.columns = ['user_id', 'night_order_ratio']
        
        # Sabah siparişleri (06-12 saatleri)
        morning_orders = orders_df.groupby('user_id').apply(
            lambda x: ((x['order_hour_of_day'] >= 6) & (x['order_hour_of_day'] < 12)).sum() / len(x)
        ).reset_index()
        morning_orders.columns = ['user_id', 'morning_order_ratio']
        
        # Öğleden sonra siparişleri (12-18 saatleri)
        afternoon_orders = orders_df.groupby('user_id').apply(
            lambda x: ((x['order_hour_of_day'] >= 12) & (x['order_hour_of_day'] < 18)).sum() / len(x)
        ).reset_index()
        afternoon_orders.columns = ['user_id', 'afternoon_order_ratio']
        
        # Tüm zaman özelliklerini birleştir
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
        Tekrar sipariş davranışı özellikleri.
        
        Özellikler:
        - overall_reorder_rate: Genel tekrar sipariş oranı.
        - avg_reorder_rate_per_order: Sipariş başına ortalama tekrar sipariş oranı.
        - reorder_consistency: Tekrar sipariş verme tutarlılığı.
        - favorite_products_count: Favori ürünlerin sayısı (5+ kez sipariş edilenler).
        """
        logger.info("Tekrar sipariş davranışı özellikleri oluşturuluyor...")
        
        # user_id'yi almak için birleştir
        order_products_with_user = order_products_df.merge(
            orders_df[['order_id', 'user_id']], 
            on='order_id'
        )
        
        # Kullanıcı başına genel tekrar sipariş oranı
        reorder_stats = order_products_with_user.groupby('user_id').agg({
            'reordered': ['mean', 'sum', 'std']
        }).reset_index()
        
        reorder_stats.columns = [
            'user_id',
            'overall_reorder_rate',
            'total_reordered_items',
            'reorder_rate_std'
        ]
        
        # Sipariş başına tekrar sipariş oranı (bazı kullanıcılar tutarlı bir şekilde tekrar sipariş verirken, diğerleri vermez)
        reorder_per_order = order_products_with_user.groupby(['user_id', 'order_id'])['reordered'].mean().reset_index()
        reorder_consistency = reorder_per_order.groupby('user_id')['reordered'].agg(['mean', 'std']).reset_index()
        reorder_consistency.columns = ['user_id', 'avg_reorder_rate_per_order', 'reorder_consistency_std']
        
        # Favori ürünler (5+ kez sipariş edilenler)
        product_order_counts = order_products_with_user.groupby(['user_id', 'product_id']).size().reset_index()
        product_order_counts.columns = ['user_id', 'product_id', 'times_ordered']
        
        favorite_products = product_order_counts[product_order_counts['times_ordered'] >= 5]\
            .groupby('user_id').size().reset_index()
        favorite_products.columns = ['user_id', 'favorite_products_count']
        
        # Birleştir
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
        Ürün çeşitliliği özellikleri.
        
        Özellikler:
        - unique_products: Benzersiz ürün sayısı.
        - unique_aisles: Benzersiz reyon sayısı.
        - unique_departments: Benzersiz departman sayısı.
        - product_diversity_score: Ürün çeşitliliği puanı.
        - avg_products_per_order: Sipariş başına ortalama ürün sayısı.
        - exploration_rate: Yeni ürünleri deneme oranı.
        """

        logger.info("Çeşitlilik özellikleri oluşturuluyor...")
        
        # Reyon ve departman bilgilerini almak için birleştir
        order_products_full = order_products_df\
            .merge(orders_df[['order_id', 'user_id', 'order_number']], on='order_id')\
            .merge(products_df[['product_id', 'aisle_id', 'department_id']], on='product_id')
        
        # Benzersiz sayımlar
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
        
        # Sipariş başına ürün sayısı
        diversity_stats['avg_products_per_order'] = (
            order_products_full.groupby('user_id').size().values / 
            diversity_stats['total_orders']
        )
        
        # Ürün çeşitliliği puanı (normalize edilmiş)
        diversity_stats['product_diversity_score'] = (
            diversity_stats['unique_products'] / 
            (diversity_stats['total_orders'] * diversity_stats['avg_products_per_order'] + 1)
        )
        
        # --- OPTİMİZE EDİLMİŞ KEŞİF ORANI HESAPLAMASI ---
        def calculate_exploration(df_group):
            if df_group.empty:
                return 0
            
            mid_point = df_group['order_number'].median()
            
            early_products = set(df_group.loc[df_group['order_number'] <= mid_point, 'product_id'])
            late_products = set(df_group.loc[df_group['order_number'] > mid_point, 'product_id'])
            
            if len(late_products) > 0:
                return len(late_products - early_products) / len(late_products)
            return 0

        # Fonksiyonu her kullanıcı grubuna uygula
        exploration_df = order_products_full.groupby('user_id').apply(calculate_exploration).reset_index(name='exploration_rate')
        
        diversity_stats = diversity_stats.merge(exploration_df, on='user_id', how='left')
        # --- OPTİMİZASYON SONU ---

        # Geçici sütunu kaldır
        diversity_stats = diversity_stats.drop('total_orders', axis=1)
        
        return diversity_stats

    def get_feature_names(self) -> List[str]:
        """Özellik adlarının bir listesini döndürür."""
        return self.feature_names


def create_behavioral_features_pipeline(orders_df: pd.DataFrame,
                                        order_products_df: pd.DataFrame,
                                        products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm davranışsal özellikleri oluşturmak için hızlı bir pipeline.
    
    Args:
        orders_df: Siparişler veri çerçevesi.
        order_products_df: Sipariş ürünleri veri çerçevesi.
        products_df: Ürünler veri çerçevesi.
        
    Returns:
        Kullanıcı düzeyinde davranışsal özelliklere sahip bir veri çerçevesi.
    """
    engineer = BehavioralFeatureEngineer()
    behavioral_features = engineer.create_all_behavioral_features(
        orders_df, order_products_df, products_df
    )
    
    return behavioral_features


# Örnek kullanım
if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.append('../../')
    from src.data.data_loader import InstacartDataLoader
    from src.config import RAW_DATA_DIR
    
    # Veriyi yükle
    loader = InstacartDataLoader(RAW_DATA_DIR)
    data = loader.load_all_data()
    
    orders_df = data['orders']
    order_products = pd.concat([
        data['order_products_prior'],
        data['order_products_train']
    ])
    products_df = data['products']
    
    # Davranışsal özellikleri oluştur
    behavioral_features = create_behavioral_features_pipeline(
        orders_df, order_products, products_df
    )
    
    print("\n Davranışsal Özellikler Örneği:")
    print(behavioral_features.head(10))
    
    print("\n Davranışsal Özellikler İstatistikleri:")
    print(behavioral_features.describe())
    
    print("\n Davranışsal özellikler başarıyla oluşturuldu!")