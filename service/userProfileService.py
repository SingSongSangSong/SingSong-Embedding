from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
import os
from dotenv import load_dotenv
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
import logging
import aiomysql
import asyncio

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class UserProfileService:
    def __init__(self):
        try:
            # Load API keys from environment
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
            
            # Initialize LLM model
            self.llm = ChatOpenAI(
                temperature=0.5,  
                max_tokens=4096,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model_name='gpt-4o-mini',  
                api_key=self.OPENAI_API_KEY
            )

            # Connect to Milvus
            connections.connect(alias="default", host=os.getenv("MILVUS_HOST", "milvus-standalone"), port="19530")
            self.collection = Collection("singsongsangsong_22286")

            # Embedding model for user profiles
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

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
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def connect_to_db(self):
        try:
            logger.info("Connecting to MySQL database asynchronously...")
            pool = await aiomysql.create_pool(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                db=os.getenv("DB_DATABASE"),
                port=int(os.getenv("DB_PORT", 3306)),
                charset='utf8mb4',
                cursorclass=aiomysql.DictCursor,
                autocommit=True
            )
            return pool
        except Exception as e:
            logger.error(f"Failed to connect to MySQL database: {e}")
            raise

    def create_user_profile_collection(self):
        try:
            if not utility.has_collection(self.user_profile_collection_name):
                logger.info(f"Creating collection {self.user_profile_collection_name} in Milvus...")
                fields = [
                    FieldSchema(name="member_id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="profile_vector", dtype=DataType.FLOAT_VECTOR, dim=3072),
                    FieldSchema(name="profile_string", dtype=DataType.VARCHAR, max_length=65535),
                ]
                schema = CollectionSchema(fields, "User profiles collection")
                self.user_profile_collection = Collection(self.user_profile_collection_name, schema)

                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "COSINE",
                    "params": {"nlist": 1024}
                }
                self.user_profile_collection.create_index(field_name="profile_vector", index_params=index_params)
            else:
                logger.info(f"Collection {self.user_profile_collection_name} already exists.")
        except Exception as e:
            logger.error(f"Failed to create or access Milvus collection: {e}")
            raise

    def insert_or_update_user_profile(self, member_id, profile_vector, profile_string):
        try:
            collection = Collection(self.user_profile_collection_name)
            collection.load()
            expr = f"member_id == {member_id}"
            results = collection.query(expr=expr, output_fields=["member_id"])

            if results:
                collection.delete(expr=expr)

            data = [[member_id], [profile_vector], [profile_string]]
            collection.insert(data)
        except Exception as e:
            logger.error(f"Failed to insert or update profile for user {member_id}: {e}")
            raise

    async def fetch_recent_member_ids(self):
        try:
            pool = await self.connect_to_db()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    query = """
                    SELECT DISTINCT member_id
                    FROM member_action
                    WHERE CREATED_AT >= DATE_SUB(NOW(), INTERVAL 1 DAY)
                    """
                    await cursor.execute(query)
                    result = await cursor.fetchall()
                    return [row['member_id'] for row in result]
        except Exception as e:
            logger.error(f"Failed to fetch recent member IDs: {e}")
            return []

    async def fetch_user_actions_for_ids(self, member_ids):
        if not member_ids:
            logger.warning("No member IDs found.")
            return {}

        try:
            pool = await self.connect_to_db()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    query = f"""
                    SELECT member_id, song_info_id, SUM(action_score) as total_score
                    FROM member_action
                    WHERE member_id IN ({','.join(['%s'] * len(member_ids))})
                    AND CREATED_AT >= DATE_SUB(NOW(), INTERVAL 3 WEEK)
                    GROUP BY member_id, song_info_id
                    """
                    await cursor.execute(query, tuple(member_ids))
                    user_actions = await cursor.fetchall()

                    user_data = {}
                    for member_id, song_id, score in user_actions:
                        if member_id not in user_data:
                            user_data[member_id] = {"song_ids": [], "scores": []}
                        user_data[member_id]["song_ids"].append(song_id)
                        user_data[member_id]["scores"].append(score)
                    return user_data
        except Exception as e:
            logger.error(f"Failed to fetch user actions: {e}")
            return {}

    def get_song_descriptions(self, song_ids):
        try:
            descriptions = []
            for song_id in song_ids:
                search_results = self.collection.query(
                    expr=f"song_info_id == {song_id}", output_fields=["description"]
                )
                descriptions.append(search_results[0]['description'] if search_results else "No description available")
            return descriptions
        except Exception as e:
            logger.error(f"Failed to fetch song descriptions: {e}")
            raise

    async def embed_user_profile(self, profile):
        try:
            return await self.embedding_model.aembed_query(profile)
        except Exception as e:
            logger.error(f"Failed to embed user profile: {e}")
            raise

    async def run(self):
        try:
            self.create_user_profile_collection()
            recent_user_ids = await self.fetch_recent_member_ids()
            user_data = await self.fetch_user_actions_for_ids(recent_user_ids)

            for member_id, user_info in user_data.items():
                user_profile = self.create_user_profile(user_info["song_ids"], user_info["scores"])
                user_embedding = await self.embed_user_profile(user_profile)
                self.insert_or_update_user_profile(member_id, user_embedding, user_profile)

            await asyncio.gather(self.create_gender_profiles())
        except Exception as e:
            logger.error(f"Error during run: {e}")
            raise