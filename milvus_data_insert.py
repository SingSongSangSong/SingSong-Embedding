import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

def safe_parse_year(value):
    try:
        if len(str(value)) == 4 and str(value).isdigit():
            return int(value)
        return pd.to_datetime(value).year
    except (ValueError, TypeError):
        return None

# Load and preprocess data
data = pd.read_csv("with_ssss_22286_updated5.csv", encoding="utf-8-sig")
song_info = pd.read_csv("song_info.csv", encoding="utf-8-sig")

# Merge song_info's is_mr value into data based on song_id and song_info_id
merged_data = pd.merge(data, song_info[['song_info_id', 'is_mr']], left_on='song_id', right_on='song_info_id', how='left')

# Convert is_mr to boolean
merged_data['MR'] = merged_data['is_mr'].astype(bool)

merged_data.dropna(subset=["song_id"], inplace=True)

# Parse year column, drop NaN values, and convert to int64
merged_data['year'] = merged_data['year'].apply(safe_parse_year)
merged_data = merged_data.dropna(subset=['year'])
merged_data['year'] = merged_data['year'].astype(np.int64)

merged_data = merged_data.rename(columns={'artist_name': 'singer_name'})

# Convert necessary fields to correct types
merged_data['song_id'] = merged_data['song_id'].astype(np.int64)
merged_data['genre'] = merged_data['genre'].astype(str)
merged_data['song_name'] = merged_data['song_name'].astype(str)
merged_data['singer_name'] = merged_data['singer_name'].astype(str)
merged_data['ssss'] = merged_data['ssss'].astype(str)

# Hugging Face model for generating embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_ssss_column(text):
    try:
        if not isinstance(text, str) or pd.isna(text):
            return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)
        
        tags = text.split(',') if ',' in text else [text]
        embeddings = model.encode(tags)
        return np.max(embeddings, axis=0).astype(np.float32) if len(embeddings) > 1 else embeddings.astype(np.float32)
    except Exception as e:
        print(f"Error processing text: {text}, Error: {e}")
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)

merged_data['ssss_embedding'] = merged_data['ssss'].apply(embed_ssss_column)

# Preprocess list columns and convert to np.float32
def parse_list_column(col):
    try:
        return np.array(eval(col), dtype=np.float32)
    except Exception as e:
        print(f"Error parsing list in column: {col}, Error: {e}")
        return np.array([], dtype=np.float32)

merged_data['mfcc_mean_parsed'] = merged_data['mfcc_mean'].apply(parse_list_column)
merged_data['chroma_stft_parsed'] = merged_data['chroma_stft'].apply(parse_list_column)
merged_data['tonnetz_parsed'] = merged_data['tonnetz'].apply(parse_list_column)
merged_data['mel_spectrogram_parsed'] = merged_data['mel_spectrogram'].apply(parse_list_column)

numerical_columns = [
    'tempo', 'harmonic_mean', 'harmonic_std', 'percussive_mean', 'percussive_std', 
    'spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'rms', 
    'dynamic_range', 'zcr', 'loudness', 'max_pitch'
]

# Ensure vectors are np.float32
def create_vector(row):
    try:
        return np.concatenate((
            row['ssss_embedding'],  # ssss embedding
            row[numerical_columns].values.astype(np.float32),  # numerical data
            row['mfcc_mean_parsed'],  # parsed list data
            row['chroma_stft_parsed'], 
            row['tonnetz_parsed'], 
            row['mel_spectrogram_parsed']
        ), axis=None)
    except Exception as e:
        print(f"Error creating vector for song_id {row['song_id']}: {e}")
        return np.zeros(model.get_sentence_embedding_dimension() + len(numerical_columns), dtype=np.float32)

merged_data['vector'] = merged_data.apply(create_vector, axis=1)

# Milvus connection
connections.connect(
    alias="default", 
    host='localhost', 
    port='19530'
)

# Milvus collection schema definition
fields = [
    FieldSchema(name="song_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=merged_data['vector'].iloc[0].shape[0]),
    FieldSchema(name="year", dtype=DataType.INT64),
    FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="song_number", dtype=DataType.INT64),
    FieldSchema(name="song_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="singer_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="MR", dtype=DataType.BOOL),
    FieldSchema(name="ssss", dtype=DataType.VARCHAR, max_length=255)
]

schema = CollectionSchema(fields, "Collection of song vectors with metadata")
collection = Collection(name="singsongsangsong_22286", schema=schema)

# Insert data into Milvus
batch_size = 100
for i in range(0, len(merged_data), batch_size):
    i_end = min(i + batch_size, len(merged_data))
    batch = merged_data.iloc[i:i_end]
    
    data = [
        batch['song_id'].tolist(),
        batch['vector'].tolist(),
        batch['year'].tolist(),
        batch['genre'].tolist(),
        batch['song_number'].tolist(),
        batch['song_name'].tolist(),
        batch['singer_name'].tolist(),
        batch['MR'].tolist(),
        batch['ssss'].tolist(),
    ]
    
    collection.insert(data)
    print(f"Inserted {i_end} documents into Milvus collection 'singsongsangsong_22286'.")

collection.flush()
print(f"Data successfully inserted and flushed to Milvus collection 'singsongsangsong_22286'.")