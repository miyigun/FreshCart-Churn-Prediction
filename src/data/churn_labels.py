"""
Müşteri Kaybı Etiketi Oluşturma Modülü (HOLD OUT STRATEJİSİ)
========================================================================
Gerçek müşteri kaybı etiketlerini tanımlamak için 'train' değerlendirme setini kullanır.
'prior' (geçmiş) ve 'train' (hedef) verilerini ayırarak Veri Sızıntısını önler.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnLabelCreator:
    """
    Instacart tarafından sağlanan 'train' değerlendirme setine dayanarak müşteri kaybı etiketleri oluşturur.
    
    Yeni Strateji (Sızıntısız):
    1. HEDEF (Etiket): 'train' setindeki satırlar, kullanıcıların BİR SONRAKİ siparişini temsil eder.
       - Eğer 'train' setindeki 'days_since_prior_order' >= churn_threshold ise -> KAYIP (1)
       - Eğer 'train' setindeki 'days_since_prior_order' < churn_threshold ise -> AKTİF (0)
       
    2. ÖZELLİKLER: SADECE 'prior' setindeki satırlardan hesaplanır.
    """
    
    def __init__(self, churn_threshold_days: int = 30):
        """
        Args:
            churn_threshold_days: Önceki siparişten bu yana geçen gün sayısı bu değere eşit veya büyükse, kullanıcı kayıp olarak kabul edilir.
                                  Not: Instacart verileri bu değeri 30 ile sınırlar, bu nedenle 30, '30+ gün' anlamına gelir.
        """
        self.churn_threshold = churn_threshold_days
        logger.info(f"Müşteri Kaybı Tanımlama Stratejisi: Sonraki Sipariş Tahmini")
        logger.info(f"Eşik Değer: days_since_prior_order >= {self.churn_threshold}")

    def create_churn_labels(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """
        SADECE 'sonraki' siparişi temsil eden 'train' seti satırlarını kullanarak etiketler oluşturur.
        
        Args:
            orders_df: Tam orders veri çerçevesi ('eval_set' sütununu içermelidir).
            
        Returns:
            ['user_id', 'is_churn', 'days_to_next_order'] sütunlarını içeren bir veri çerçevesi.
        """
        logger.info("'train' seti hedefi kullanılarak müşteri kaybı etiketleri oluşturuluyor...")
        
        # 1. Yalnızca 'train' seti satırlarını filtrele. Bunlar bizim hedeflerimiz.
        # Not: 'test' seti satırlarının etiketi yoktur (Kaggle gönderimi için), bu yüzden burada onları yok sayıyoruz.
        train_targets = orders_df[orders_df['eval_set'] == 'train'].copy()
        
        if train_targets.empty:
            logger.error("orders_df içinde 'train' satırı bulunamadı! Verilerin doğru yüklendiğinden emin olun.")
            raise ValueError("'train' değerlendirme seti bulunamadı.")

        # 2. Hedef Değişkeni Tanımla
        # NaN değerlerini işle (ilk siparişler train setinde olmamalı, ama güvenlik için iyidir)
        train_targets['days_since_prior_order'] = train_targets['days_since_prior_order'].fillna(0)
        
        # Etiket oluştur: kayıp ise 1 (>= 30 gün), aktif ise 0 (< 30 gün)
        train_targets['is_churn'] = (
            train_targets['days_since_prior_order'] >= self.churn_threshold
        ).astype(int)
        
        # 3. İlgili sütunları tut
        # Analiz için 'days_since_prior_order' sütununu 'days_to_next_order' olarak saklıyoruz
        labels_df = train_targets[['user_id', 'is_churn', 'days_since_prior_order']].rename(
            columns={'days_since_prior_order': 'days_to_next_order'}
        )
        
        # 4. İstatistikler
        self._print_stats(labels_df)
        
        return labels_df

    def _print_stats(self, labels_df: pd.DataFrame):
        """Etiket dağılım istatistiklerini yazdırmak için yardımcı fonksiyon."""
        total_users = len(labels_df)
        churn_cnt = labels_df['is_churn'].sum()
        active_cnt = total_users - churn_cnt
        churn_rate = churn_cnt / total_users
        
        logger.info(f"\n{'='*80}")
        logger.info(f"MÜŞTERİ KAYBI ETİKET İSTATİSTİKLERİ (Gerçek Değerler)")
        logger.info(f"{'='*80}")
        logger.info(f"Toplam Hedef Kullanıcı:      {total_users:>10,}")
        logger.info(f"Kaybedilen (>=30 gün):     {churn_cnt:>10,} ({churn_rate:.2%})")
        logger.info(f"Aktif (<30 gün):       {active_cnt:>10,} ({1-churn_rate:.2%})")
        logger.info(f"{'='*80}\n")

    def split_train_test_stratified(self, 
                                    master_df: pd.DataFrame, 
                                    test_size: float = 0.2,
                                    random_state: int = 42):
        """
        Nihai veri setinde katmanlı train-test ayrımı gerçekleştirir.
        Etiketler için veri setinin kendi 'train' ayrımına güvendiğimizden,
        modelimizi doğrulamak için burada kullanıcıları rastgele ayırıyoruz.
        """
        from sklearn.model_selection import train_test_split
        
        logger.info(f"Veri ayrılıyor (Test boyutu: {test_size}, Katmanlı)...")
        
        X = master_df.drop(['user_id', 'is_churn', 'eval_set', 'days_to_next_order'], axis=1, errors='ignore')
        y = master_df['is_churn']
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# Mantığı test etmek için örnek kullanım
if __name__ == "__main__":
    # Test mantığı için sahte veri oluşturma
    # Gerçek kullanımda, gerçek verileri yükleyin
    print("ChurnLabelCreator mantığı test ediliyor...")
    
    mock_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'eval_set': ['train', 'train', 'train', 'prior', 'test'],
        'days_since_prior_order': [30.0, 7.0, 14.0, 5.0, 0.0]
    })
    
    creator = ChurnLabelCreator(churn_threshold_days=30)
    labels = creator.create_churn_labels(mock_data)
    
    print("\nSonuç Etiketleri:")
    print(labels)
    
    # Beklenen: 
    # Kullanıcı 1: Kayıp (30 >= 30)
    # Kullanıcı 2: Aktif (7 < 30)
    # Kullanıcı 3: Aktif (14 < 30)
    # Kullanıcı 4: Yok sayıldı (prior)
    # Kullanıcı 5: Yok sayıldı (test)