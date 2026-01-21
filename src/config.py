from dataclasses import dataclass

@dataclass
class Config:
    # Data
    train_path: str = "data/total_sentence_data/total_sentence_data/total_sentence_train.csv"
    test_path: str = "data/total_sentence_data/total_sentence_data/total_sentence_test.csv"
    max_length: int = 256
    
    # Model
    text_encoder: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-d, fast
    product_type_emb_dim: int = 64
    hidden_dims: list = None
    dropout: float = 0.1
    
    # Training
    batch_size: int = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 2
    warmup_ratio: float = 0.1
    
    # System
    num_workers: int = 4
    seed: int = 42
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 64]
