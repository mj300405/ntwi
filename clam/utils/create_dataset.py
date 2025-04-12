import os
import pandas as pd
from pathlib import Path

def create_dataset_csv(split_type):
    """
    Create CSV for either train or test split
    Args:
        split_type (str): Either 'train' or 'test'
    """
    # Lists to store data
    slides = []
    labels = []
    
    # Process data from the specified split
    data_dir = Path(split_type)
    
    # Define directory names based on split type
    if split_type == 'train':
        pos_dir_name = 'EGFR_positive'
        neg_dir_name = 'EGFR_negative'
    else:  # test
        pos_dir_name = 'C-S-EGFR_positive'
        neg_dir_name = 'C-S-EGFR_negative'
    
    # Process EGFR positive cases
    pos_dir = data_dir / pos_dir_name
    if pos_dir.exists():
        for slide_dir in pos_dir.glob('*'):
            if slide_dir.is_dir() and not slide_dir.name.startswith('.'):
                # Verify that directory contains tiles
                if list(slide_dir.glob('*.png')):
                    slides.append(slide_dir.name)
                    labels.append(1)  # 1 for EGFR positive
            
    # Process EGFR negative cases
    neg_dir = data_dir / neg_dir_name
    if neg_dir.exists():
        for slide_dir in neg_dir.glob('*'):
            if slide_dir.is_dir() and not slide_dir.name.startswith('.'):
                # Verify that directory contains tiles
                if list(slide_dir.glob('*.png')):
                    slides.append(slide_dir.name)
                    labels.append(0)  # 0 for EGFR negative
    
    # Create DataFrame
    df = pd.DataFrame({
        'slide_id': slides,
        'label': labels
    })
    
    # Create directory if it doesn't exist
    os.makedirs('dataset_csv', exist_ok=True)
    
    # Save to CSV
    csv_path = f'dataset_csv/egfr_{split_type}.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n{split_type.capitalize()} set statistics:")
    print(f"Total slides: {len(df)}")
    print(f"EGFR positive: {sum(labels)}")
    print(f"EGFR negative: {len(labels) - sum(labels)}")
    print(f"CSV file saved to: {csv_path}")

if __name__ == '__main__':
    # Create CSVs for both train and test sets
    create_dataset_csv('train')
    create_dataset_csv('test') 