"""
Veri Yükleme Modülü
===================
Instacart veri setini yüklemek ve birleştirmek için fonksiyonlar.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging
from tqdm import tqdm

# Günlük kaydı (logging) kurulumu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstacartDataLoader:
    """Instacart verilerini yüklemek için sınıf."""
    
    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: Ham veri dosyalarının bulunduğu dizin.
        """
        self.data_dir = Path(data_dir)
        self.data = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Tüm Instacart CSV dosyalarını yükler.
        
        Returns:
            Tüm veri çerçevelerini içeren bir sözlük.
        """
        logger.info("Instacart veri setleri yükleniyor...")
        
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
            logger.info(f"Yükleniyor: {filename}...")
            
            try:
                self.data[key] = pd.read_csv(filepath)
                logger.info(f"Yüklendi {key}: {self.data[key].shape}")
            except FileNotFoundError:
                logger.error(f"Dosya bulunamadı: {filepath}")
                raise
            except Exception as e:
                logger.error(f"{filename} yüklenirken hata oluştu: {str(e)}")
                raise
        
        logger.info(f"Tüm veri setleri başarıyla yüklendi!\n")
        self._print_data_summary()
        
        return self.data
    
    def _print_data_summary(self):
        """Yüklenen verilerin bir özetini yazdırır."""
        logger.info("=" * 80)
        logger.info("VERİ ÖZETİ")
        logger.info("=" * 80)
        
        for name, df in self.data.items():
            logger.info(f"{name:25s}: {df.shape[0]:>10,} satır x {df.shape[1]:>3} sütun")
            logger.info(f"{'':25s}  Bellek: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        logger.info("=" * 80 + "\n")
    
    def merge_order_products(self) -> pd.DataFrame:
        """
        'prior' ve 'train' order_products veri çerçevelerini birleştirir.
        
        Returns:
            Birleştirilmiş order_products veri çerçevesi.
        """
        logger.info("order_products veri setleri birleştiriliyor...")
        
        order_products = pd.concat([
            self.data['order_products_prior'],
            self.data['order_products_train']
        ], ignore_index=True)
        
        logger.info(f"Birleştirilmiş order_products: {order_products.shape}")
        
        return order_products
    
    def create_master_dataset(self) -> pd.DataFrame:
        """
        Tüm tabloları birleştirerek ana bir veri seti oluşturur.
        
        Returns:
            Tüm bilgileri içeren ana bir veri çerçevesi.
        """
        logger.info("Ana veri seti oluşturuluyor...")
        
        # order_products birleştirme
        order_products = self.merge_order_products()
        
        # Ürün bilgilerini ekleme
        logger.info("Ürünler ile birleştiriliyor...")
        df = order_products.merge(
            self.data['products'],
            on='product_id',
            how='left'
        )
        
        # Reyon bilgilerini ekleme
        logger.info("Reyonlar ile birleştiriliyor...")
        df = df.merge(
            self.data['aisles'],
            on='aisle_id',
            how='left'
        )
        
        # Departman bilgilerini ekleme
        logger.info("Departmanlar ile birleştiriliyor...")
        df = df.merge(
            self.data['departments'],
            on='department_id',
            how='left'
        )
        
        # Sipariş bilgilerini ekleme
        logger.info("Siparişler ile birleştiriliyor...")
        df = df.merge(
            self.data['orders'],
            on='order_id',
            how='left'
        )
        
        logger.info(f"Ana veri seti oluşturuldu: {df.shape}")
        logger.info(f"Sütunlar: {list(df.columns)}\n")
        
        return df
    
    def get_user_order_history(self, user_id: int) -> pd.DataFrame:
        """
        Belirli bir kullanıcının sipariş geçmişini getirir.
        
        Args:
            user_id: Kullanıcı ID'si.
            
        Returns:
            Kullanıcının sipariş geçmişini bir veri çerçevesi olarak döndürür.
        """
        master_df = self.create_master_dataset()
        user_history = master_df[master_df['user_id'] == user_id].copy()
        
        logger.info(f"Kullanıcı {user_id} geçmişi: {user_history.shape[0]} sipariş")
        
        return user_history
    
    def get_data_info(self) -> Dict:
        """
        Veri hakkında detaylı bilgi sağlar.
        
        Returns:
            Veri istatistiklerini içeren bir sözlük.
        """
        if not self.data:
            logger.warning("Henüz veri yüklenmedi!")
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
        İşlenmiş veriyi kaydeder.
        
        Args:
            df: Kaydedilecek veri çerçevesi.
            filename: Çıktı dosya adı.
            output_dir: Çıktı dizini (varsayılan: data/processed).
        """
        if output_dir is None:
            output_dir = self.data_dir.parent / "processed"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        
        logger.info(f"Kaydediliyor: {filepath}...")
        
        if filepath.suffix == '.parquet':
            df.to_parquet(filepath, index=False)
        elif filepath.suffix == '.csv':
            df.to_csv(filepath, index=False)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {filepath.suffix}")
        
        logger.info(f"Başarıyla kaydedildi!")


def load_instacart_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Hızlı bir yükleyici fonksiyonu.
    
    Args:
        data_dir: Instacart CSV dosyalarını içeren dizin.
        
    Returns:
        Veri çerçevelerini içeren bir sözlük.
    """
    loader = InstacartDataLoader(data_dir)
    return loader.load_all_data()


def create_sample_data(data: Dict[str, pd.DataFrame], 
                      sample_size: int = 10000) -> Dict[str, pd.DataFrame]:
    """
    Hızlı test için örnek veri oluşturur.
    
    Args:
        data: Orijinal veri sözlüğü.
        sample_size: Örneklenecek kullanıcı sayısı.
        
    Returns:
        Örneklenmiş bir veri sözlüğü.
    """
    logger.info(f"{sample_size} kullanıcı ile örnek veri seti oluşturuluyor...")
    
    # Kullanıcıları örnekle
    sample_users = data['orders']['user_id'].drop_duplicates().sample(
        n=min(sample_size, data['orders']['user_id'].nunique()),
        random_state=42
    )
    
    # Siparişleri filtrele
    sample_orders = data['orders'][
        data['orders']['user_id'].isin(sample_users)
    ].copy()
    
    sample_order_ids = sample_orders['order_id'].unique()
    
    # order_products filtrele
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
    
    logger.info(f"Örnek veri seti oluşturuldu:")
    logger.info(f"Kullanıcılar: {len(sample_users):,}")
    logger.info(f"Siparişler: {len(sample_orders):,}")
    logger.info(f"Siparişlerdeki ürünler: {len(sample_op_prior) + len(sample_op_train):,}\n")
    
    return sampled_data


# Örnek kullanım
if __name__ == "__main__":
    from pathlib import Path
    
    # Örnek: Veri yükleme
    data_dir = Path("../../data/raw")
    
    if data_dir.exists():
        loader = InstacartDataLoader(data_dir)
        data = loader.load_all_data()
        
        # Ana veri setini oluştur
        master_df = loader.create_master_dataset()
        
        # İşlenmiş veriyi kaydet
        loader.save_processed_data(master_df, "master_dataset.parquet")
        
        # Bilgileri yazdır
        info = loader.get_data_info()
        print("\n Veri Seti İstatistikleri:")
        for key, value in info.items():
            print(f"   {key}: {value}")
    else:
        print(f"Veri dizini bulunamadı: {data_dir}")
        print("Lütfen önce Instacart verilerini indirin!")