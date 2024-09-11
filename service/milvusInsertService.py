import pandas as pd
import numpy as np
from langchain_openai import OpenAIEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility, MilvusException
import time
import logging
import openai
import math
from openai import OpenAI


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusInsertService:
    def __init__(self, collection_name, host='localhost', port='19530'):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None

    def connect_milvus(self, retries=5, delay=10):
        """Milvus에 연결하는 함수"""
        for attempt in range(retries):
            try:
                connections.connect(alias="default", host=self.host)
                logger.info(f"Connected to Milvus at {self.host}")
                return
            except MilvusException as e:
                logger.info(f"Failed to connect to Milvus (attempt {attempt+1}/{retries}): {e}")
                time.sleep(delay)
        raise Exception(f"Could not connect to Milvus after {retries} attempts")

    def create_collection_if_not_exists(self, dim):
        """Milvus 컬렉션이 존재하는지 확인하고, 없으면 생성"""
        if not utility.has_collection(self.collection_name):
            logger.info(f"Collection '{self.collection_name}' does not exist. Creating collection.")
            
            fields = [
                FieldSchema(name="song_info_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="year", dtype=DataType.INT64),
                FieldSchema(name="genre", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="country", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="singer_type", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="audio_file_url", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="song_number", dtype=DataType.INT64),
                FieldSchema(name="song_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="artist_name", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="MR", dtype=DataType.BOOL),
                FieldSchema(name="ssss", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="album", dtype=DataType.VARCHAR, max_length=255)
            ]
            
            schema = CollectionSchema(fields, "Collection of song vectors with metadata")
            self.collection = Collection(name=self.collection_name, schema=schema)
            logger.info(f"Collection '{self.collection_name}' created.")

            # 인덱스 생성
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="vector", index_params=index_params)
            logger.info(f"Index created for vector field in collection '{self.collection_name}'.")

            # 데이터 변경 사항을 플러시
            self.collection.flush()  
        else:
            self.collection = Collection(name=self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists.")

    def check_collection_entity_count(self):
        """컬렉션의 엔티티 수 확인"""
        self.collection.load()
        entity_count = self.collection.num_entities
        logger.info(f"Collection '{self.collection.name}' has {entity_count} entities.")
        return entity_count

    def safe_parse_year(self, value):
        """년도를 안전하게 파싱"""
        try:
            if len(str(value)) == 4 and str(value).isdigit():
                return int(value)
            return pd.to_datetime(value).year
        except (ValueError, TypeError):
            return None

    def parse_list_column(self, col):
        """리스트 열을 np.float32 배열로 변환"""
        try:
            return np.array(eval(col), dtype=np.float32)
        except Exception as e:
            logger.info(f"Error parsing list in column: {col}, Error: {e}")
            return np.array([], dtype=np.float32)

    def load_and_preprocess_data(self, song_info_path, data_path):
        """데이터를 로드하고 전처리"""
        song_info = pd.read_csv(song_info_path, encoding="utf-8-sig")
        data = pd.read_csv(data_path, encoding="utf-8-sig")

        merged_data = pd.merge(data, song_info[['song_info_id', 'is_mr']], left_on='song_id', right_on='song_info_id', how='left')
        merged_data['MR'] = merged_data['is_mr'].astype(bool)
        merged_data.dropna(subset=["song_id"], inplace=True)
        merged_data['year'] = merged_data['year'].apply(self.safe_parse_year).astype('Int64', errors='ignore')

        # Parse necessary list fields
        merged_data['mfcc_mean_parsed'] = merged_data['mfcc_mean'].apply(self.parse_list_column)
        merged_data['chroma_stft_parsed'] = merged_data['chroma_stft'].apply(self.parse_list_column)
        merged_data['mel_spectrogram_parsed'] = merged_data['mel_spectrogram'].apply(self.parse_list_column)

        # Filter required columns
        columns_to_save = [
            'song_number', 'song_name', 'artist_name', 'album', 'genre', 'year', 
            'country', 'singer_type', 'audio_file_url', 'max_pitch', 'ssss', 'MR', 
            'song_info_id_x', 'tempo', 'harmonic_mean', 'percussive_mean', 'spectral_centroid', 'rms',
            'mfcc_mean_parsed', 'chroma_stft_parsed', 'mel_spectrogram_parsed'
        ]
        filtered_data = merged_data[columns_to_save].copy()  # Create an explicit copy to avoid chained assignment
        filtered_data.rename(columns={'song_info_id_x': 'song_info_id'}, inplace=True)

        # Use .loc to avoid SettingWithCopyWarning
        filtered_data.loc[:, 'year'] = filtered_data['year'].apply(lambda x: int(x) if not pd.isna(x) else None).astype('Int64')
        filtered_data.loc[:, 'year'].fillna(0, inplace=True)
        filtered_data.loc[:, 'genre'].fillna("", inplace=True)
        filtered_data.loc[:, 'country'].fillna("", inplace=True)
        filtered_data.loc[:, 'singer_type'].fillna("", inplace=True)
        filtered_data.loc[:, 'album'].fillna("", inplace=True)

        return filtered_data
    
    # Function to convert year into decade format (e.g., 2010s for 2011)
    def format_decade(self, year):
        if pd.notnull(year):
            decade = (year // 10) * 10  # Get the decade
            return f"{decade}"
        return None

    # Function to convert frequency (Hz) to note and octave
    def frequency_to_note_octave(self, frequency):
        # Constants
        A4 = 440.0  # Frequency of A4 in Hz
        NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        if frequency <= 0:
            return "Unknown pitch"

        # Calculate the number of semitones away from A4
        semitone_diff = int(round(12 * np.log2(frequency / A4)))
        
        # Determine octave and note
        octave = 4 + (semitone_diff // 12)
        note = NOTES[semitone_diff % 12]
        
        return f"{note}{octave}"
    
    # Function to describe MFCC data based on mean and standard deviation
    def mfcc_description(self, mfcc_data):
        if len(mfcc_data) == 0:
            return "No MFCC data available."
        
        # Calculate mean and standard deviation
        mean_mfcc = np.mean(mfcc_data[:5])  # First 5 values
        std_mfcc = np.std(mfcc_data[:5])
        
        if mean_mfcc < -100:
            mean_desc = "The song has a very soft and mellow timbre."
        elif mean_mfcc < 0:
            mean_desc = "The song has a soft timbre with moderate complexity."
        else:
            mean_desc = "The song has a bright and complex timbre."
        
        if std_mfcc < 20:
            std_desc = "The timbre remains relatively stable throughout the song."
        else:
            std_desc = "The timbre varies significantly, adding complexity to the sound."
        
        return f"{mean_desc} {std_desc}"

    # Function to describe Chroma STFT data based on mean and standard deviation
    def chroma_stft_description(self, chroma_data):
        if len(chroma_data) == 0:
            return "No Chroma STFT data available."
        
        # Calculate mean and standard deviation
        mean_chroma = np.mean(chroma_data[:12])
        std_chroma = np.std(chroma_data[:12])
        
        if mean_chroma < 0.2:
            mean_desc = "The song has a simple harmonic structure."
        elif mean_chroma < 0.5:
            mean_desc = "The song has a moderately rich harmonic structure."
        else:
            mean_desc = "The song exhibits a complex harmonic structure."
        
        if std_chroma < 0.1:
            std_desc = "The harmonic structure remains consistent."
        else:
            std_desc = "The harmonic structure varies significantly, providing richness."
        
        return f"{mean_desc} {std_desc}"

    # Function to describe Mel-Spectrogram data based on mean and standard deviation
    def mel_spectrogram_description(self, mel_data):
        if len(mel_data) == 0:
            return "No Mel-Spectrogram data available."
        
        # Calculate mean and standard deviation
        mean_mel = np.mean(mel_data[:10])
        std_mel = np.std(mel_data[:10])
        
        if mean_mel < 1:
            mean_desc = "The song has low energy with a calm sound."
        elif mean_mel < 5:
            mean_desc = "The song has moderate energy with balanced sound textures."
        else:
            mean_desc = "The song has high energy with dynamic sound textures."
        
        if std_mel < 2:
            std_desc = "The energy level remains stable."
        else:
            std_desc = "The energy level fluctuates, adding excitement to the song."
        
        return f"{mean_desc} {std_desc}"

    # Function to generate song description based on numeric and audio features
    def generate_description(self, row):
        singer = row['artist_name'] if pd.notnull(row['artist_name']) else None
        song = row['song_name'] if pd.notnull(row['song_name']) else None
        year = self.format_decade(row['year']) if pd.notnull(row['year']) else None  # Format year as decade
        genre = row['genre'] if pd.notnull(row['genre']) else None
        singer_type = row['singer_type'] if pd.notnull(row['singer_type']) else None
        country = row['country'] if pd.notnull(row['country']) else None
        tempo = row['tempo_explanation'] if pd.notnull(row['tempo_explanation']) else None
        ssss = row['ssss'] if pd.notnull(row['ssss']) else None
        max_pitch = self.frequency_to_note_octave(row['max_pitch']) if pd.notnull(row['max_pitch']) else None

        # Parse audio feature data
        mfcc_desc = self.mfcc_description(row['mfcc_mean_parsed'])
        chroma_stft_desc = self.chroma_stft_description(row['chroma_stft_parsed'])
        mel_spectrogram_desc = self.mel_spectrogram_description(row['mel_spectrogram_parsed'])

        description = ""

        if singer and song:
            description += f"{singer} released the song '{song}'"
        elif singer:
            description += f"{singer} released a song"
        elif song:
            description += f"The song '{song}'"

        if year:
            description += f" in the {year}s. "

        if max_pitch:
            description += f"The song has a maximum pitch of {max_pitch}. "

        if genre:
            description += f"The genre of this song is {genre}. "

        if ssss:
            description += f"The song has {ssss}."

        if singer_type:
            description += f"{singer_type.lower()}"

        if country:
            if singer_type:
                description += f" from {country}. "
            else:
                description += f"From {country}. "

        if tempo:
            description += f"{tempo} "

        # Add numeric feature explanations
        description += row['harmonic_mean_explanation'] + " "
        description += row['percussive_mean_explanation'] + " "
        description += row['spectral_centroid_explanation'] + " "
        description += row['rms_explanation']

        # Add MFCC, Chroma STFT, and Mel-Spectrogram explanations
        description += mfcc_desc + " "
        description += chroma_stft_desc + " "
        description += mel_spectrogram_desc
        
        return description.strip()
    
    # Function to assign text labels based on 5-level classification
    def assign_text_label(self, value, min_val, max_val):
        """Assigns a text label based on the value in comparison to min and max"""
        normalized_value = (value - min_val) / (max_val - min_val) if max_val > min_val else 0  # Min-max scaling
        
        if normalized_value <= 0.2:
            return "Very Low"
        elif normalized_value <= 0.4:
            return "Low"
        elif normalized_value <= 0.6:
            return "Medium"
        elif normalized_value <= 0.8:
            return "High"
        else:
            return "Very High"
        

    # Function to generate explanations for general numeric features
    def general_numeric_explanation(self, feature_name, label):
        explanations = {
            "tempo": {
                "Very Low": "The song has a very slow tempo, giving it a relaxed and calm feel.",
                "Low": "The song has a slow tempo, which creates a laid-back atmosphere.",
                "Medium": "The song has a moderate tempo, balancing between calm and energetic.",
                "High": "The song has a fast tempo, adding excitement and energy.",
                "Very High": "The song has a very fast tempo, making it lively and full of energy."
            },
            "harmonic_mean": {
                "Very Low": "The harmonic structure is minimal, contributing to a clean sound.",
                "Low": "The harmonic structure is simple, giving the song a straightforward tone.",
                "Medium": "The harmonic structure is moderately rich, adding some depth.",
                "High": "The harmonic structure is quite complex, enriching the song.",
                "Very High": "The harmonic structure is highly complex, providing a deep and intricate sound."
            },
            "percussive_mean": {
                "Very Low": "The percussive elements are barely noticeable, creating a soft, mellow rhythm.",
                "Low": "The percussive elements are subtle, maintaining a gentle rhythm.",
                "Medium": "The percussive elements are balanced, contributing to a steady and clear rhythm.",
                "High": "The percussive elements are prominent, giving the song a strong and defined rhythm.",
                "Very High": "The percussive elements are highly dominant, driving the rhythm with intensity."
            },
            "spectral_centroid": {
                "Very Low": "The song’s brightness is very low, contributing to a darker and softer sound.",
                "Low": "The song has low brightness, giving it a mellow, warm tone.",
                "Medium": "The song has moderate brightness, balancing between warm and bright tones.",
                "High": "The song is fairly bright, adding sharpness and clarity to the sound.",
                "Very High": "The song is very bright, delivering a sharp and vibrant tone."
            },
            "rms": {
                "Very Low": "The song has a very low root-mean-square (RMS) level, contributing to a soft, delicate sound.",
                "Low": "The RMS level is low, giving the song a quiet and restrained dynamic.",
                "Medium": "The RMS level is moderate, balancing between soft and loud elements.",
                "High": "The RMS level is high, providing a louder, more energetic sound.",
                "Very High": "The RMS level is very high, creating an intense and powerful dynamic."
            }
        }
        return explanations.get(feature_name, {}).get(label, "No Data")
    
        

    # Function to classify and explain numeric values for 5 levels
    def classify_and_explain_numeric(self, row, column, min_val, max_val):
        value = row[column]
        label = self.assign_text_label(value, min_val, max_val)
        return self.general_numeric_explanation(column, label)
    
    # Apply classification to numeric fields and generate explanations
    def apply_numeric_classification(self, merged_data):
        # Get min and max for numeric fields
        numeric_columns = ['tempo', 'harmonic_mean', 'percussive_mean', 
                        'spectral_centroid', 'rms']
        
        for column in numeric_columns:
            min_val = merged_data[column].min()
            max_val = merged_data[column].max()
            merged_data[column + '_label'] = merged_data.apply(lambda row: self.assign_text_label(row[column], min_val, max_val), axis=1)
            merged_data[column + '_explanation'] = merged_data.apply(lambda row: self.classify_and_explain_numeric(row, column, min_val, max_val), axis=1)
        return merged_data

    def generate_embeddings(self, merged_data, client, batch_size=100, delay_seconds=1):
        """Generate embeddings using OpenAI API with batching."""
        # Apply numeric classification and generate descriptions
        merged_data = self.apply_numeric_classification(merged_data)
        merged_data['description'] = merged_data.apply(self.generate_description, axis=1)
        
        def get_batch_embeddings(text_list):
            try:
                # Ensure input to the API is just plain text (list of strings)
                response = client.embeddings.create(
                    model="text-embedding-3-large",
                    input=text_list
                )
                # Extract embeddings from response
                return [embedding.embedding for embedding in response.data]
            except Exception as e:
                # Log any errors encountered during the embedding creation process
                print(f"Error during embedding creation: {e}")
                raise e

        # Batch processing to handle large datasets
        num_batches = math.ceil(len(merged_data) / batch_size)
        all_embeddings = []

        for i in range(num_batches):
            # Get batch of descriptions
            batch_descriptions = merged_data['description'].iloc[i * batch_size: (i + 1) * batch_size].tolist()
            
            # Make sure the batch contains only strings and is not empty
            if batch_descriptions:
                print(f"Processing batch {i+1}/{num_batches}")
                batch_embeddings = get_batch_embeddings(batch_descriptions)
                all_embeddings.extend(batch_embeddings)
            
            # Pause for rate limit handling
            print(f"Pausing for {delay_seconds} seconds to avoid rate limits...")
            time.sleep(delay_seconds)

        # Add embeddings to the merged_data DataFrame
        merged_data['vector'] = all_embeddings

        return merged_data

    def insert_into_milvus(self, merged_data, batch_size=100):
        """Milvus에 데이터 삽입"""
        for i in range(0, len(merged_data), batch_size):
            batch = merged_data.iloc[i:i+batch_size]
            data = [
                batch['song_info_id'].tolist(),
                batch['vector'].tolist(),
                batch['year'].tolist(),
                batch['genre'].tolist(),
                batch['country'].tolist(),
                batch['singer_type'].tolist(),
                batch['audio_file_url'].tolist(),
                batch['song_number'].tolist(),
                batch['song_name'].tolist(),
                batch['artist_name'].tolist(),
                batch['MR'].tolist(),
                batch['ssss'].tolist(),
                batch['description'].tolist(),
                batch['album'].tolist(),
            ]
            self.collection.insert(data)
            logger.info(f"Inserted {i + len(batch)} documents into Milvus.")

    def run(self, song_info_path, data_path):
        """Milvus에 데이터를 삽입하는 전체 프로세스 실행"""
        self.connect_milvus()

        # 컬렉션 생성 또는 로드
        self.create_collection_if_not_exists(dim=3072)  # 임베딩 모델의 벡터 크기에 맞춤

        # 컬렉션에 20,000개 이상의 엔티티가 있는지 확인
        entity_count = self.check_collection_entity_count()

        if entity_count == 0:
            logger.info(f"No entities found in the collection. Proceeding with data loading and insertion.")
            
            # 데이터 로드 및 전처리
            merged_data = self.load_and_preprocess_data(song_info_path, data_path)
            
            # 임베딩 모델 로드
            client = OpenAI()

            # 임베딩 생성
            merged_data = self.generate_embeddings(merged_data, client)

            # 데이터 삽입
            self.insert_into_milvus(merged_data)
            self.collection.flush()
            logger.info(f"Data successfully inserted and flushed to Milvus collection '{self.collection_name}'.")
        else:
            logger.info(f"Collection '{self.collection_name}' already has {entity_count} entities, skipping data insertion.")