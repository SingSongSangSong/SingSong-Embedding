import pymysql
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from sentence_transformers import SentenceTransformer
from langchain_milvus.vectorstores import Milvus
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class UserProfileService:
    def __init__(self):
        # Load API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

        # Initialize LLM model
        self.llm = ChatOpenAI(
            temperature=0.5,  # 창의성 (0.0 ~ 2.0)
            max_tokens=4096,  # 최대 토큰수
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_name='gpt-4o-mini',  # 모델명
            api_key=self.OPENAI_API_KEY
        )

        # Connect to MySQL and Milvus
        self.db_connection = self.connect_to_db()
        connections.connect(alias="default", host="localhost", port="19530")
        self.collection = Collection("singsongsangsong_22286")

        # Embedding model for user profiles
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Initialize prompt template for profile creation
        self.prompt_template = """
            Based on the following song descriptions and user scores, create a detailed user profile. Ensure the profile is written as a single, cohesive paragraph, adhering strictly to the following format:

            1. **Genres:** Describe the genres the user prefers.
            2. **Time Period:** Mention the time periods (decades) the user prefers.
            3. **Country:** Indicate the country or region of the songs/artists the user enjoys.
            4. **Artist Type:** Highlight whether the user prefers solo artists, groups, or both.
            5. **Artist Gender:** Highlight whether the user prefers male, female, or both.
            6. **Musical Characteristics:** Focus on key musical features (e.g., tempo, energy, mood, timbre, harmonic structure) that the user consistently enjoys.
            7. **Tags:** Highlight key tags (e.g., breakup, nostalgia, energetic) the user often prefers in songs.

            Here are the user's song descriptions and scores:

            Songs:
            {song_descriptions}

            Scores:
            {song_scores}

            Profile:
        """
        self.prompt = PromptTemplate.from_template(template=self.prompt_template)
        self.llm_chain = LLMChain(llm=self.llm, prompt=self.prompt)

        # Ensure the user profile collection exists in Milvus
        self.user_profile_collection_name = "user_profile"

    # MySQL 연결 설정
    def connect_to_db(self):
        logger.info("Connecting to MySQL database...")
        return pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_DATABASE")
        )

    # Insert or update a user profile in the user_profile collection
    def insert_or_update_user_profile(self, user_id, profile_vector, profile_string, recommended_songs, song_descriptions):
        collection = Collection(self.user_profile_collection_name)
        collection.load()

        # Check if the user profile already exists
        expr = f"user_id == {user_id}"
        results = collection.query(expr=expr, output_fields=["user_id"])

        if results:
            # If the profile exists, delete the old one before inserting the new one
            logger.info(f"Updating existing profile for user {user_id}.")
            collection.delete(expr=expr)

        # Convert recommended_songs and song_descriptions lists to comma-separated strings
        recommended_songs_str = ",".join(map(str, recommended_songs))  # Convert list to comma-separated string
        song_descriptions_str = ",".join(song_descriptions)  # Convert list to comma-separated string

        # Insert the new profile
        data = [
            [user_id],                  # user_id
            [profile_vector],            # profile_vector
            [profile_string],            # profile_string
            [recommended_songs_str],     # recommended_songs
            [song_descriptions_str]      # song_descriptions
        ]
        collection.insert(data)
        logger.info(f"Profile for user {user_id} inserted/updated successfully.")

    # Normalize action scores to a range of 1 to 10
    def normalize_scores(self, scores):
        min_score = min(scores)
        max_score = max(scores)

        # If all scores are the same, just return 5s (since normalizing doesn't make sense)
        if min_score == max_score:
            return [5] * len(scores)

        # Normalize scores to a 1-10 range
        normalized_scores = [
            1 + ((score - min_score) / (max_score - min_score)) * 9
            for score in scores
        ]
        return normalized_scores

    # 최근 하루 동안 활동한 유저 ID 조회
    def fetch_recent_user_ids(self):
        logger.info("Fetching recent user IDs from the database...")
        cursor = self.db_connection.cursor()

        # 최근 하루 동안 활동한 유저들의 ID 조회
        query = """
        SELECT DISTINCT member_id
        FROM member_action
        WHERE CREATED_AT >= DATE_SUB(NOW(), INTERVAL 1 DAY)
        """
        
        cursor.execute(query)
        recent_user_ids = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        logger.info(f"Fetched {len(recent_user_ids)} recent user IDs.")
        return recent_user_ids

    # 특정 유저 ID들에 해당하는 song_id와 action_score를 조회
    def fetch_user_actions_for_ids(self, user_ids):
        logger.info(f"Fetching user actions for {len(user_ids)} users from the database...")
        cursor = self.db_connection.cursor()

        # 특정 유저 ID들에 대한 song_id와 action_score 조회
        query = """
        SELECT member_id, song_info_id, SUM(action_score) as total_score
        FROM member_action
        WHERE member_id IN (%s)
        GROUP BY member_id, song_info_id
        """ % ','.join(['%s'] * len(user_ids))
        
        cursor.execute(query, tuple(user_ids))
        user_actions = cursor.fetchall()
        
        user_data = {}
        for user_id, song_id, score in user_actions:
            if user_id not in user_data:
                user_data[user_id] = {"song_ids": [], "scores": []}
            user_data[user_id]["song_ids"].append(song_id)
            user_data[user_id]["scores"].append(score)
        
        cursor.close()
        logger.info(f"Fetched actions for {len(user_data)} users.")
        return user_data

    # Milvus에서 song_id와 description 가져오기
    def get_song_descriptions(self, song_ids):
        logger.info(f"Fetching song descriptions for {len(song_ids)} songs from Milvus...")
        descriptions = []
        for song_id in song_ids:
            search_results = self.collection.query(
                expr=f"song_info_id == {song_id}", 
                output_fields=["description"]
            )
            if search_results:
                descriptions.append(search_results[0]['description'])
            else:
                descriptions.append("No description available")
        logger.info(f"Fetched descriptions for {len(descriptions)} songs.")
        return descriptions

    # 사용자 프로파일 생성 (최대 10곡까지만)
    def create_user_profile(self, user_song_ids, user_scores):
        logger.info("Creating user profile based on song descriptions and user scores...")

        # Normalize the scores to a 1-10 range
        normalized_scores = self.normalize_scores(user_scores)

        # Fetch the song descriptions based on song_ids
        song_descriptions = self.get_song_descriptions(user_song_ids)

        # Limit to 10 songs/descriptions, sorted by the normalized scores
        top_songs_and_scores = sorted(zip(song_descriptions, normalized_scores), key=lambda x: x[1], reverse=True)[:10]

        # Unzip the top 10 songs and scores
        top_songs, top_scores = zip(*top_songs_and_scores)

        # Format the song descriptions and scores into a numbered list
        song_descriptions_text = "\n".join([f"{i+1}. {desc}" for i, desc in enumerate(top_songs)])
        song_scores_text = "\n".join([f"{i+1}. {score:.2f}" for i, score in enumerate(top_scores)])

        # Pass the song descriptions and scores into the LLM chain to generate the user profile
        user_profile = self.llm_chain.run({
            "song_descriptions": song_descriptions_text,
            "song_scores": song_scores_text
        })
        
        logger.info("User profile created.")
        return user_profile

    # 유저 프로파일 임베딩
    def embed_user_profile(self, profile):
        logger.info("Embedding the user profile...")
        embedding = self.embedding_model.encode(profile)  # 유저 프로파일 임베딩
        logger.info("User profile embedded.")
        return embedding

    # Milvus에서 유사한 노래 검색
    def search_similar_songs(self, user_embedding, top_k=20):
        logger.info("Searching for similar songs in Milvus...")
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        
        user_embedding = np.expand_dims(user_embedding, axis=0).astype(np.float32)
        
        search_results = self.collection.search(
            data=user_embedding,  # 사용자 임베딩 벡터
            anns_field="vector",  # 벡터 필드 이름
            param=search_params,
            limit=top_k,
            output_fields=["song_info_id", "description"]
        )
        
        logger.info(f"Found {len(search_results)} similar songs.")
        return search_results

    # 전체 프로세스를 실행하는 함수
    def run(self):
        logger.info("Starting user profile creation and similar song search process...")

        # Fetch recent user IDs
        recent_user_ids = self.fetch_recent_user_ids()

        # Fetch user actions for the fetched recent user IDs
        user_data = self.fetch_user_actions_for_ids(recent_user_ids)

        for user_id, user_info in user_data.items():
            # 1. 유저 프로파일 생성
            logger.info(f"Processing user {user_id}...")
            user_profile = self.create_user_profile(user_info["song_ids"], user_info["scores"])

            # 2. 유저 프로파일 임베딩
            user_embedding = self.embed_user_profile(user_profile)

            # 3. 유사한 노래 검색
            similar_songs = self.search_similar_songs(user_embedding)
            recommended_song_ids = []
            song_descriptions = []

            logger.info(f"User {user_id} - Similar Songs:")

            # Iterate over each result, which contains a list of hits
            for result in similar_songs:
                for hit in result:
                    song_info_id = hit.id  # Extract song ID
                    distance = hit.distance  # Extract distance (similarity score)
                    description = hit.entity.description  # Extract song description
                    
                    recommended_song_ids.append(song_info_id)  # Add song ID to recommendations
                    song_descriptions.append(description)  # Add description to list
                    logger.info(f"Song ID: {song_info_id}, Distance: {distance}, Description: {description}")

            # 4. Insert or update user profile in Milvus, including song descriptions
            self.insert_or_update_user_profile(
                user_id=user_id,
                profile_vector=user_embedding.tolist(),  # Convert to list before storing
                profile_string=user_profile,
                recommended_songs=recommended_song_ids,
                song_descriptions=song_descriptions  # Pass the descriptions
            )

        logger.info("User profile creation and song search process completed.")