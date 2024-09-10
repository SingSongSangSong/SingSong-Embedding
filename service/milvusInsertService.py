import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, Collection, DataType, utility, MilvusException
import time
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusInsertService:
    def __init__(self, collection_name, host='milvus-standalone', port='19530'):
        self.collection_name = collection_name
        self.host = host
        self.port = port
        self.collection = None

    def connect_milvus(self, retries=5, delay=10):
        """Milvus에 연결하는 함수"""
        for attempt in range(retries):
            try:
                # 포트를 생략하고 호스트만으로 연결
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
        # Load the CSV data
        song_info = pd.read_csv(song_info_path, encoding="utf-8-sig")
        data = pd.read_csv(data_path, encoding="utf-8-sig")

        # Merge the song_info and data on 'song_id'
        merged_data = pd.merge(data, song_info[['song_info_id', 'is_mr']], left_on='song_id', right_on='song_info_id', how='left')
        
        # Process the 'MR' column
        merged_data['MR'] = merged_data['is_mr'].astype(bool)
        
        # Drop rows where 'song_id' is missing
        merged_data.dropna(subset=["song_id"], inplace=True)

        # Parse the 'year' column
        merged_data['year'] = merged_data['year'].apply(self.safe_parse_year).astype('Int64', errors='ignore')

        # Parse necessary list fields (e.g., 'mfcc_mean', 'chroma_stft', 'mel_spectrogram')
        merged_data['mfcc_mean_parsed'] = merged_data['mfcc_mean'].apply(self.parse_list_column)
        merged_data['chroma_stft_parsed'] = merged_data['chroma_stft'].apply(self.parse_list_column)
        merged_data['mel_spectrogram_parsed'] = merged_data['mel_spectrogram'].apply(self.parse_list_column)

        # Filter required columns for final CSV output
        columns_to_save = [
            'song_number', 'song_name', 'artist_name', 'album', 'genre', 'year', 
            'country', 'singer_type', 'audio_file_url', 'max_pitch', 'ssss', 'MR', 
            'song_info_id_x'
        ]

        # Create a new DataFrame with the selected columns
        filtered_data = merged_data[columns_to_save]

        # Rename the 'song_info_id_x' column to 'song_info_id'
        filtered_data.rename(columns={'song_info_id_x': 'song_info_id'}, inplace=True)

        # Handle missing values for 'year', 'genre', 'country', 'singer_type', 'album'
        filtered_data['year'] = filtered_data['year'].apply(lambda x: int(x) if not pd.isna(x) else None).astype('Int64')
        filtered_data['year'].fillna(0, inplace=True)
        filtered_data['genre'].fillna("", inplace=True)
        filtered_data['country'].fillna("", inplace=True)
        filtered_data['singer_type'].fillna("", inplace=True)
        filtered_data['album'].fillna("", inplace=True)

        return filtered_data

    def generate_description(self, row):
        """노래 설명 생성"""
        description = ""
        description += f"{row['artist_name']} released the song '{row['song_name']}'" if pd.notnull(row['artist_name']) and pd.notnull(row['song_name']) else ""
        description += f" in the {row['year']}s." if pd.notnull(row['year']) else ""
        description += f" The genre of this song is {row['genre']}." if pd.notnull(row['genre']) else ""
        description += f" The song has {row['ssss']}." if pd.notnull(row['ssss']) else ""
        return description.strip()

    def generate_embeddings(self, merged_data, model):
        """임베딩 생성과 동시에 description 생성"""
        merged_data['description'] = merged_data.apply(self.generate_description, axis=1)
        merged_data['vector'] = merged_data['description'].apply(lambda x: model.encode(str(x)))
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
        self.create_collection_if_not_exists(dim=384)  # 임의로 384를 사용하지만 실제 벡터 크기는 임베딩 모델이 결정

        # 컬렉션에 20,000개 이상의 엔티티가 있는지 확인
        entity_count = self.check_collection_entity_count()

        if entity_count == 0:
            # 컬렉션에 엔티티가 없으면 데이터 로드 및 임베딩 생성
            logger.info(f"No entities found in the collection. Proceeding with data loading and insertion.")
            
            # 데이터 로드 및 전처리
            merged_data = self.load_and_preprocess_data(song_info_path, data_path)
            
            # 임베딩 모델 로드
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            # 임베딩 생성
            merged_data = self.generate_embeddings(merged_data, model)

            # 데이터 삽입
            self.insert_into_milvus(merged_data)
            self.collection.flush()
            logger.info(f"Data successfully inserted and flushed to Milvus collection '{self.collection_name}'.")
        else:
            logger.info(f"Collection '{self.collection_name}' already has {entity_count} entities, skipping data insertion.")