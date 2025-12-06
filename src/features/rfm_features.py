"""
RFM Özellik Mühendisliği Modülü
================================
Yenilik (Recency), Sıklık (Frequency) ve Parasal (Monetary) özellikleri oluşturur.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RFMFeatureEngineer:
    """
    RFM (Yenilik, Sıklık, Parasal) özellikleri oluşturur.
    
    Özellikler:
    - Yenilik (Recency): Son siparişten bu yana geçen gün sayısı.
    - Sıklık (Frequency): Sipariş sıklığı.
    - Parasal (Monetary): Parasal değer (sepet büyüklüğünü vekil olarak kullanarak).
    """
    
    def __init__(self):
        self.feature_names = []
    
    def create_all_rfm_features(self, 
                                orders_df: pd.DataFrame,
                                order_products_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm RFM özelliklerini oluşturur.
        
        Args:
            orders_df: Siparişler veri çerçevesi.
            order_products_df: Sipariş ürünleri veri çerçevesi.
            
        Returns:
            Kullanıcı düzeyinde RFM özelliklerine sahip bir veri çerçevesi.
        """
        logger.info("RFM özellikleri oluşturuluyor...")
        
        # Yenilik özellikleri
        recency_features = self.create_recency_features(orders_df)
        
        # Sıklık özellikleri
        frequency_features = self.create_frequency_features(orders_df)
        
        # Parasal özellikler (sepet büyüklüğünü vekil olarak kullanarak)
        monetary_features = self.create_monetary_features(orders_df, order_products_df)
        
        # Hepsini birleştir
        rfm_features = recency_features\
            .merge(frequency_features, on='user_id', how='outer')\
            .merge(monetary_features, on='user_id', how='outer')
        
        # NaN değerlerini 0 ile doldur
        rfm_features = rfm_features.fillna(0)
        
        self.feature_names = [col for col in rfm_features.columns if col != 'user_id']
        
        logger.info(f"{len(self.feature_names)} adet RFM özelliği oluşturuldu")
        logger.info(f"Özellikler: {self.feature_names}")
        
        return rfm_features
    
    def create_recency_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        Yenilik özellikleri.
        
        Özellikler:
        - days_since_last_order: Son siparişten bu yana geçen gün sayısı.
        - days_since_first_order: İlk siparişten bu yana geçen gün sayısı.
        - customer_age_days: Müşteri yaşı (gün olarak).
        """
        logger.info("Yenilik özellikleri oluşturuluyor...")
        
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
        
        # Genel maksimum sipariş numarası (referans noktası - "şimdi")
        global_max = orders_df['order_number'].max()
        
        # Yenilik hesaplamaları
        user_recency['orders_since_last'] = global_max - user_recency['last_order_number']
        user_recency['days_since_last_order'] = user_recency['orders_since_last'] * 7  # Tahmin
        
        user_recency['total_order_span'] = user_recency['last_order_number'] - user_recency['first_order_number']
        user_recency['customer_age_days'] = user_recency['total_order_span'] * 7  # Tahmin
        
        # İlk siparişten bu yana geçen günler
        user_recency['days_since_first_order'] = user_recency['customer_age_days'] + user_recency['days_since_last_order']
        
        # Nihai özellikleri seç
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
        Sıklık özellikleri.
        
        Özellikler:
        - total_orders: Toplam sipariş sayısı.
        - orders_per_day: Günlük ortalama sipariş sayısı.
        - order_frequency: Sipariş sıklığı puanı.
        - order_regularity: Sipariş düzenliliği (düşük standart sapma daha düzenli demektir).
        """
        logger.info("Sıklık özellikleri oluşturuluyor...")
        
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
        
        # Türetilmiş özellikler
        user_frequency['order_span'] = user_frequency['last_order_number'] - user_frequency['first_order_number']
        user_frequency['estimated_customer_days'] = user_frequency['order_span'] * 7
        
        # Günlük sipariş sayısı (sıklık oranı)
        user_frequency['orders_per_day'] = user_frequency['total_orders'] / (user_frequency['estimated_customer_days'] + 1)
        
        # Sipariş düzenliliği (değişim katsayısı)
        user_frequency['order_regularity'] = (
            user_frequency['std_days_between_orders'] / 
            (user_frequency['avg_days_between_orders'] + 1)
        )
        
        # Standart sapmadaki NaN değerlerini doldur (sadece 1-2 sipariş olduğunda olur)
        user_frequency['std_days_between_orders'] = user_frequency['std_days_between_orders'].fillna(0)
        user_frequency['order_regularity'] = user_frequency['order_regularity'].fillna(0)
        
        # Nihai özellikleri seç
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
        Parasal özellikler.
        
        Not: Fiyat bilgisi olmadığı için sepet büyüklüğünü vekil olarak kullanıyoruz.
        
        Özellikler:
        - avg_basket_size: Ortalama sepet büyüklüğü (ürün sayısı).
        - total_products_ordered: Sipariş edilen toplam ürün sayısı.
        - avg_unique_products: Sipariş başına ortalama benzersiz ürün.
        - basket_size_std: Sepet büyüklüğünün değişkenliği.
        """
        logger.info("Parasal özellikler oluşturuluyor (sepet büyüklüğünü vekil olarak kullanarak)...")
        
        # Sipariş başına sepet büyüklüğünü hesapla
        basket_sizes = order_products_df.groupby('order_id').agg({
            'product_id': ['count', 'nunique']
        }).reset_index()
        
        basket_sizes.columns = ['order_id', 'basket_size', 'unique_products_in_order']
        
        # user_id'yi almak için siparişlerle birleştir
        baskets_with_user = orders_df[['order_id', 'user_id']].merge(
            basket_sizes, on='order_id', how='left'
        )
        
        # Kullanıcı düzeyinde toplama
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
        
        # NaN değerlerini doldur
        user_monetary['basket_size_std'] = user_monetary['basket_size_std'].fillna(0)
        
        # Sepet büyüklüğü tutarlılığı (düşük olması daha tutarlı demektir)
        user_monetary['basket_size_cv'] = (
            user_monetary['basket_size_std'] / 
            (user_monetary['avg_basket_size'] + 1)
        )
        
        # Nihai özellikleri seç
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
        RFM skorunu hesaplar (1-5 arası bir ölçekte).
        
        RFM Skoru = Yenilik Skoru + Sıklık Skoru + Parasal Skoru
        Yüksek bir skor, değerli bir müşteriyi gösterir.
        
        Args:
            rfm_features: RFM özellikleri veri çerçevesi.
            
        Returns:
            RFM skorlarını içeren bir veri çerçevesi.
        """
        logger.info("RFM skorları hesaplanıyor...")
        
        rfm_scored = rfm_features.copy()
        
        # Yenilik skoru (düşük olması daha iyidir, bu yüzden etiketleri ters çeviriyoruz)
        rfm_scored['recency_score'] = pd.qcut(
            rfm_scored['days_since_last_order'], 
            q=5, 
            labels=[5, 4, 3, 2, 1],
            duplicates='drop'
        )
        
        # Sıklık skoru (yüksek olması daha iyidir)
        rfm_scored['frequency_score'] = pd.qcut(
            rfm_scored['total_orders'], 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        )
        
        # Parasal skor (yüksek olması daha iyidir)
        rfm_scored['monetary_score'] = pd.qcut(
            rfm_scored['avg_basket_size'], 
            q=5, 
            labels=[1, 2, 3, 4, 5],
            duplicates='drop'
        )
        
        # Tam sayıya dönüştür
        rfm_scored['recency_score'] = rfm_scored['recency_score'].astype(int)
        rfm_scored['frequency_score'] = rfm_scored['frequency_score'].astype(int)
        rfm_scored['monetary_score'] = rfm_scored['monetary_score'].astype(int)
        
        # Genel RFM skoru
        rfm_scored['rfm_score'] = (
            rfm_scored['recency_score'] + 
            rfm_scored['frequency_score'] + 
            rfm_scored['monetary_score']
        )
        
        # RFM segmenti (basitleştirilmiş)
        rfm_scored['rfm_segment'] = pd.cut(
            rfm_scored['rfm_score'],
            bins=[0, 6, 9, 12, 15],
            labels=['Risk Altında', 'Umut Veren', 'Sadık', 'Şampiyonlar']
        )
        
        logger.info(f"RFM skorları hesaplandı")
        logger.info(f"\nRFM Segment Dağılımı:")
        print(rfm_scored['rfm_segment'].value_counts().sort_index())
        
        return rfm_scored
    
    def get_feature_names(self) -> List[str]:
        """Özellik adlarının bir listesini döndürür."""
        return self.feature_names


def create_rfm_features_pipeline(orders_df: pd.DataFrame,
                                 order_products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Tüm RFM özelliklerini oluşturmak için hızlı bir işlem hattı (pipeline).
    
    Args:
        orders_df: Siparişler veri çerçevesi.
        order_products_df: Sipariş ürünleri veri çerçevesi.
        
    Returns:
        Kullanıcı düzeyinde RFM özellikleri ve skorları içeren bir veri çerçevesi.
    """
    engineer = RFMFeatureEngineer()
    
    # Özellikleri oluştur
    rfm_features = engineer.create_all_rfm_features(orders_df, order_products_df)
    
    # RFM skorlarını ekle
    rfm_with_scores = engineer.create_rfm_score(rfm_features)
    
    return rfm_with_scores


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
    
    # RFM özelliklerini oluştur
    rfm_features = create_rfm_features_pipeline(orders_df, order_products)
    
    print("\nRFM Özellikleri Örneği:")
    print(rfm_features.head(10))
    
    print("\nRFM Özellikleri İstatistikleri:")
    print(rfm_features.describe())
    
    print("\nRFM özellikleri başarıyla oluşturuldu!")