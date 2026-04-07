"""
compute_stats.py – Compute the exact mean and std used during model training.
Run this ONCE from the project root directory:

    cd "c:/Users/MOHAMMED/Documents/silent doctor"
    python chatbot_project/compute_stats.py

This outputs two numbers you can copy into model.py.
"""

import os
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.model_selection import train_test_split

base_skin_dir = os.path.join('.', 'input')

imageid_path_dict = {
    os.path.splitext(os.path.basename(x))[0]: x
    for x in glob(os.path.join(base_skin_dir, '*', '*.jpg'))
}

lesion_type_dict = {
    'nv': 'Melanocytic nevi', 'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ', 'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses', 'vasc': 'Vascular lesions', 'df': 'Dermatofibroma'
}

print("Loading metadata …")
skin_df = pd.read_csv(os.path.join(base_skin_dir, 'HAM10000_metadata.csv'))
skin_df['path']          = skin_df['image_id'].map(imageid_path_dict.get)
skin_df['cell_type']     = skin_df['dx'].map(lesion_type_dict.get)
skin_df['cell_type_idx'] = pd.Categorical(skin_df['cell_type']).codes
skin_df['age']           = skin_df['age'].fillna(skin_df['age'].mean())

print("Loading images (this may take a few minutes) …")
skin_df['image'] = skin_df['path'].map(
    lambda x: np.asarray(Image.open(x).resize((100, 75)))
)

features = skin_df.drop(columns=['cell_type_idx'])
target   = skin_df['cell_type_idx']

x_train_o, _, _, _ = train_test_split(features, target, test_size=0.20, random_state=1234)

x_train = np.asarray(x_train_o['image'].tolist(), dtype=np.float32)

TRAIN_MEAN = float(np.mean(x_train))
TRAIN_STD  = float(np.std(x_train))

print("\n" + "="*50)
print(f"TRAIN_MEAN = {TRAIN_MEAN:.6f}")
print(f"TRAIN_STD  = {TRAIN_STD:.6f}")
print("="*50)
print("\n✅ Copy these values into chatbot_project/model.py")
print("   Replace the TRAIN_MEAN and TRAIN_STD lines.")
