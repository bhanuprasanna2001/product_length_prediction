import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pytorch_lightning as pl
from transformers import AutoTokenizer


class ProductDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, product_type_map, is_test=False):
        self.texts = df['TOTAL_SENTENCE'].tolist()
        self.product_types = df['PRODUCT_TYPE_ID'].map(product_type_map).fillna(0).astype(int).tolist()
        self.targets = None if is_test else df['PRODUCT_LENGTH'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'product_type': torch.tensor(self.product_types[idx], dtype=torch.long)
        }
        
        if not self.is_test:
            item['target'] = torch.tensor(self.targets[idx], dtype=torch.float32)
        
        return item


class ProductDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
        self.product_type_map = None
        
    def setup(self, stage=None):
        train_df = pd.read_csv(self.config.train_path)
        
        # Build product type mapping
        all_types = train_df['PRODUCT_TYPE_ID'].unique()
        self.product_type_map = {t: i+1 for i, t in enumerate(all_types)}  # 0 reserved for unknown
        self.num_product_types = len(all_types) + 1
        
        # Split train/val
        val_size = int(0.1 * len(train_df))
        self.train_df = train_df.iloc[:-val_size]
        self.val_df = train_df.iloc[-val_size:]
        
        self.train_ds = ProductDataset(
            self.train_df, self.tokenizer, self.config.max_length, self.product_type_map
        )
        self.val_ds = ProductDataset(
            self.val_df, self.tokenizer, self.config.max_length, self.product_type_map
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=False,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=False,
            persistent_workers=True if self.config.num_workers > 0 else False
        )
