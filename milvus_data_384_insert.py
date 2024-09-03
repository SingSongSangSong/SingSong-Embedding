import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

def hz_to_midi(hz: float):
    try:
        return 69 + 12 * np.log2(hz / 440.0)
    except Exception as e:
        raise Exception(f"Error converting Hz to MIDI: {e}")

def midi_to_note_octave(midi: float):
    try:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        note = int(midi) % 12
        octave = int(midi // 12) - 1
        return note_names[note], octave
    except Exception as e:
        raise Exception(f"Error converting MIDI to note and octave: {e}")

def convert_max_pitch_to_note_octave(max_pitch: float):
    try:
        if pd.notna(max_pitch):
            midi_note = hz_to_midi(max_pitch)
            note, octave = midi_to_note_octave(midi_note)
            return f"{note}{octave}"
        else:
            return None
    except Exception as e:
        raise Exception(f"Error converting max pitch to note and octave: {e}")


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

def create_item_preference_sentence(row):
    try:
        item_preferences_parts = []

        if pd.notna(row['genre']):
            item_preferences_parts.append(f"This song belongs to the {row['genre']} genre.")
        
        if pd.notna(row['year']):
            decade = (row['year'] // 10) * 10
            item_preferences_parts.append(f"It was released in the {decade}s.")

        if pd.notna(row['country']):
            item_preferences_parts.append(f"It originates from {row['country']}.")

        if pd.notna(row['singer_type']):
            item_preferences_parts.append(f"It is performed by a {row['singer_type']}.")

        if pd.notna(row['ssss']):
            item_preferences_parts.append(f"The song has a {row['ssss']} mood.")

        if pd.notna(row['max_pitch']):
            note_octave = convert_max_pitch_to_note_octave(row['max_pitch'])
            item_preferences_parts.append(f"It has a max pitch of {note_octave}.")

        return " ".join(item_preferences_parts)
    except Exception as e:
        raise Exception(f"Error creating item preference sentence: {e}")

def embed_ssss_column(row):
    try:
        # Create a descriptive sentence for the item (song)
        item_sentence = create_item_preference_sentence(row)
        
        # Embed the generated sentence
        embedding = model.encode(item_sentence).astype(np.float32)
        
        return embedding
    except Exception as e:
        print(f"Error processing row with song_id {row['song_id']}, Error: {e}")
        return np.zeros(model.get_sentence_embedding_dimension(), dtype=np.float32)

merged_data['ssss_embedding'] = merged_data.apply(embed_ssss_column, axis=1)

# Milvus connection
connections.connect(
    alias="default", 
    host='localhost', 
    port='19530'
)

# Milvus collection schema definition
fields = [
    FieldSchema(name="song_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=merged_data['ssss_embedding'].iloc[0].shape[0]),
    FieldSchema(name="year", dtype=DataType.INT64),
    FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="song_number", dtype=DataType.INT64),
    FieldSchema(name="song_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="singer_name", dtype=DataType.VARCHAR, max_length=255),
    FieldSchema(name="MR", dtype=DataType.BOOL),
    FieldSchema(name="ssss", dtype=DataType.VARCHAR, max_length=255)
]

schema = CollectionSchema(fields, "Collection of song vectors with metadata")
collection = Collection(name="singsongsangsong_22286_size_384", schema=schema)

# Insert data into Milvus
batch_size = 100
for i in range(0, len(merged_data), batch_size):
    i_end = min(i + batch_size, len(merged_data))
    batch = merged_data.iloc[i:i_end]
    
    data = [
        batch['song_id'].tolist(),
        batch['ssss_embedding'].tolist(),
        batch['year'].tolist(),
        batch['genre'].tolist(),
        batch['song_number'].tolist(),
        batch['song_name'].tolist(),
        batch['singer_name'].tolist(),
        batch['MR'].tolist(),
        batch['ssss'].tolist(),
    ]
    
    collection.insert(data)
    print(f"Inserted {i_end} documents into Milvus collection 'singsongsangsong_22286_size_384'.")

collection.flush()
print(f"Data successfully inserted and flushed to Milvus collection 'singsongsangsong_22286_size_384'.")