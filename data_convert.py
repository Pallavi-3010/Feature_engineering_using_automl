"""
convert_arff_to_csv.py - Convert ARFF to CSV
"""

from scipy.io import arff
import pandas as pd
import os

def convert_arff_to_csv(arff_file, csv_file=None):
    """
    Convert ARFF file to CSV
    
    Args:
        arff_file: Path to .arff file
        csv_file: Output CSV path (optional)
    """
    print(f"Converting {arff_file} to CSV...")
    
    # Load ARFF file
    data, meta = arff.loadarff(arff_file)
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Decode byte strings to normal strings (ARFF quirk)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.decode('utf-8')
            except:
                pass
    
    # Auto-generate CSV filename if not provided
    if csv_file is None:
        csv_file = arff_file.replace('.arff', '.csv')
    
    # Save to CSV
    df.to_csv(csv_file, index=False)
    
    print(f"✓ Converted successfully!")
    print(f"  Output: {csv_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    
    return csv_file


if __name__ == "__main__":
    # Convert Madelon
    if os.path.exists(r'C:\My_projects\AutoML\phpfLuQE4.arff'):
        convert_arff_to_csv(r'C:\My_projects\AutoML\phpfLuQE4.arff', r'C:\My_projects\AutoML\phpfLuQE4.csv')
        print("\n✓ Now you can use madelon.csv with your preprocessing script!")
    else:
        print("Error: madelon.arff not found in datasets/")