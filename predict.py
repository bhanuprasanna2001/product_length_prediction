import torch
import pandas as pd
from tqdm import tqdm

from src.config import Config
from src.data.dataset import ProductDataset
from src.model import ProductLengthModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader


def predict(checkpoint_path: str, output_path: str = "submission.csv"):
    config = Config()
    
    # Load model
    model = ProductLengthModel.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda() if torch.cuda.is_available() else model.cpu()
    
    # Load test data
    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)
    
    all_types = train_df['PRODUCT_TYPE_ID'].unique()
    product_type_map = {t: i+1 for i, t in enumerate(all_types)}
    
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
    test_ds = ProductDataset(test_df, tokenizer, config.max_length, product_type_map, is_test=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size * 2, shuffle=False, num_workers=4)
    
    predictions = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            pred = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['product_type'].to(device)
            )
            predictions.extend(pred.cpu().numpy().tolist())
    
    # Create submission
    submission = pd.DataFrame({
        'PRODUCT_ID': test_df['PRODUCT_ID'],
        'PRODUCT_LENGTH': predictions
    })
    submission.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    import sys
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best.ckpt"
    predict(checkpoint)
