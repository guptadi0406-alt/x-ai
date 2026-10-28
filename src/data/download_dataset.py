"""
Dataset download helper.

Downloads the Airline Passenger Satisfaction dataset.
If Kaggle API is not available, provides manual download instructions.
"""

import os
import sys
from pathlib import Path


def download_dataset():
    """Attempt to download or provide instructions."""
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'
    data_dir.mkdir(exist_ok=True)
    
    target_file = data_dir / 'airline_satisfaction.csv'
    
    if target_file.exists():
        print(f"Dataset already exists at: {target_file}")
        return str(target_file)
    
    # Try kaggle API
    try:
        import kaggle
        print("Downloading from Kaggle...")
        kaggle.api.dataset_download_files(
            'teejmahal20/airline-passenger-satisfaction',
            path=str(data_dir),
            unzip=True
        )
        # Find downloaded files
        for f in data_dir.iterdir():
            if f.suffix == '.csv' and 'train' in f.name.lower():
                f.rename(target_file)
                print(f"Dataset saved to: {target_file}")
                return str(target_file)
    except ImportError:
        pass
    except Exception as e:
        print(f"Kaggle download failed: {e}")
    
    # Provide manual instructions
    print("""
╔══════════════════════════════════════════════════════════════════╗
║           DATASET DOWNLOAD INSTRUCTIONS                        ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  The Airline Passenger Satisfaction dataset is needed.          ║
║                                                                ║
║  Option 1: Download from Kaggle                                ║
║    https://www.kaggle.com/datasets/teejmahal20/                ║
║    airline-passenger-satisfaction                               ║
║                                                                ║
║  Option 2: Use kaggle CLI                                      ║
║    pip install kaggle                                           ║
║    kaggle datasets download -d teejmahal20/                    ║
║      airline-passenger-satisfaction -p data/ --unzip            ║
║                                                                ║
║  After downloading, place the CSV file as:                     ║
║    data/airline_satisfaction.csv                                ║
║                                                                ║
║  Or place train.csv/test.csv from the Kaggle download          ║
║  directly into the data/ directory.                            ║
║                                                                ║
╚══════════════════════════════════════════════════════════════════╝
""")
    return None


if __name__ == '__main__':
    download_dataset()
