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
from openai import AsyncOpenAI
from service.functionCallingPrompts import ExtractCommonTraitService, ExtractCommonTraits
from pydantic import BaseModel
from typing import List

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 로드
load_dotenv()

class ExtractCommonTraitsForUserProfile(BaseModel):
    genre: List[str]
    year: List[str]
    country: List[str]
    artist_type: List[str]
    artist_gender: List[str]
    situation: List[str]
    octave: List[str]
    vocal_range: List[str]
    lyrics: List[str]
    artist_name: List[str]

class UserProfileService:
    def __init__(self):
        try:
            # Load API keys from environment
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
            self.asyncOpenai = AsyncOpenAI(api_key=self.OPENAI_API_KEY)
            
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
            self.collection = Collection("final_song_embeddings")

            self.extractService = ExtractCommonTraitService(self.asyncOpenai)

            # Embedding model for user profiles
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

            # Initialize prompt template for profile creation
            self.prompt_template = """
                Based on the following song descriptions, scores, and user preferences from their previous interactions, create a detailed and personalized user profile. This profile should be designed in a structured format that can be directly used to generate tailored song recommendations. Ensure that the profile reflects the user's consistent patterns, highlighting key preferences, and providing insights based on the user's song history.

                Consider the following:
                1. Analyze all genres, time periods, and countries of the songs the user has interacted with.
                2. Note the user's tendency towards specific artist types, genders, or song characteristics.
                3. Pay attention to recurring musical elements (e.g., high-energy songs, emotional ballads, dance beats) and patterns in their song scores.
                4. Examine octave or vocal range preferences and highlight if the user leans towards songs with a specific vocal range or style.
                5. Identify any situational tags or specific moods (e.g., breakup, nostalgia, happy) associated with the user's song choices.

                **Format**:
                - Genre: [<genre>] (if applicable, otherwise `[]`)
                - Year: [<year>] (if applicable, otherwise `[]`)
                - Country: [<country>] (if applicable, otherwise `[]`)
                - Artist Type: [<artist_type>] (if applicable, otherwise `[]`)
                - Artist Gender: [<artist_gender>] (if applicable, otherwise `[]`)
                - Situation: [<situation>] (if applicable, otherwise [])
                - Octave: [<octave>]  (if applicable, otherwise `[]`)
                - Vocal Range: [<vocal_range>] (if applicable, otherwise `[]`)
                - Artist Name: [<artist_name>] (If applicable, otherwise `[]`)
                - Lyrics: [<lyrics>] (if applicable, otherwise `[]`)

                Here are the user's song descriptions and scores:

                **Songs**:
                {song_descriptions}

                **Scores**:
                {song_scores}

                **Profile**:
            """

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
                    for row in user_actions:
                        member_id, song_id, score = row['member_id'], row['song_info_id'], row['total_score']
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
            search_results = self.collection.query(
                expr=f"song_info_id in {song_ids}", output_fields=["description"]
            )
            descriptions = [result['description'] for result in search_results]
            return descriptions
        except Exception as e:
            logger.error(f"Failed to fetch song descriptions: {e}")
            raise

    # Normalize action scores to a range of 1 to 10
    def normalize_scores(self, scores):
        try:
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
        except Exception as e:
            logger.error(f"Failed to normalize scores: {e}")
            raise

    async def embed_user_profile(self, profile):
        try:
            return await self.embedding_model.aembed_query(profile)
        except Exception as e:
            logger.error(f"Failed to embed user profile: {e}")
            raise
    
    # 사용자 프로파일 생성 (최대 10곡까지만)
    async def create_user_profile(self, user_song_ids, user_scores):
        try:
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

            # Create the prompt text by filling in the song descriptions and scores into the template
            prompt_filled = self.prompt_template.format(song_descriptions=song_descriptions_text, song_scores=song_scores_text)

            # Step 6: Use the LLM to refine the query
            response = await self.asyncOpenai.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt_filled}],
                response_format=ExtractCommonTraitsForUserProfile,
            )
            parsed_result = response.choices[0].message.parsed

            # Formatting each field into the desired string format
            formatted_output = f"""
                GENRE: {', '.join(parsed_result.genre)}
                YEAR: {', '.join(parsed_result.year)}
                COUNTRY: {', '.join(parsed_result.country)}
                ARTIST TYPE: {', '.join(parsed_result.artist_type)}
                ARTIST GENDER: {', '.join(parsed_result.artist_gender)}
                SITUATION: {', '.join(parsed_result.situation)}
                OCTAVE: {', '.join(parsed_result.octave)}
                VOCAL RANGE: {', '.join(parsed_result.vocal_range)}
                LYRICS: {', '.join(parsed_result.lyrics)}
                ARITST NAME: {', '.join(parsed_result.artist_name)}
            """
            
            logger.info("User profile created.")
            return formatted_output
        except Exception as e:
            logger.error(f"Failed to create user profile: {e}")
            raise
    
    # 성별에 따른 전체 성향 조회
    async def fetch_gender_based_actions(self, gender=None):
        try:
            logger.info(f"Fetching actions for {'all' if gender is None else gender} users from the database...")
            pool = await self.connect_to_db()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # 성별에 따른 모든 유저의 song_id와 action_score를 조회 (가정: member 테이블에 gender 필드 존재)
                    if gender is None:
                        query = """
                        SELECT song_info_id, SUM(action_score) as total_score
                        FROM member_action
                        JOIN member ON member_action.member_id = member.member_id
                        GROUP BY song_info_id
                        """
                        await cursor.execute(query)
                    else:
                        query = """
                        SELECT song_info_id, SUM(action_score) as total_score
                        FROM member_action
                        JOIN member ON member_action.member_id = member.member_id
                        WHERE member.gender = %s
                        GROUP BY song_info_id
                        """
                        await cursor.execute(query, (gender,))
                    actions = await cursor.fetchall()
            
            song_ids = []
            scores = []

            for songs in actions:
                song_ids.append(songs['song_info_id'])
                scores.append(songs['total_score'])
            
            logger.info(f"Fetched {len(song_ids)} songs for {'all' if gender is None else gender} users.")
            return song_ids, scores
        except Exception as e:
            logger.error(f"Failed to fetch gender-based actions: {e}")
            raise

    # 남성과 여성 전체 성향 벡터 생성 및 저장
    async def create_gender_profiles(self):
        try:
            # 남성 전체 성향 (아이디 0)
            logger.info("Creating profile for all male users...")
            male_song_ids, male_scores = await self.fetch_gender_based_actions("MALE")
            male_profile = await self.create_user_profile(male_song_ids, male_scores)
            male_embedding = await self.embed_user_profile(male_profile)
            self.insert_or_update_user_profile(
                member_id=0,  # 남성 아이디 0
                profile_vector=male_embedding,
                profile_string=male_profile,
            )

            # 여성 전체 성향 (아이디 -1)
            logger.info("Creating profile for all female users...")
            female_song_ids, female_scores = await self.fetch_gender_based_actions("FEMALE")
            female_profile = await self.create_user_profile(female_song_ids, female_scores)
            female_embedding = await self.embed_user_profile(female_profile)
            self.insert_or_update_user_profile(
                member_id=-1,  # 여성 아이디 -1
                profile_vector=female_embedding,
                profile_string=female_profile,
            )

            # 전체 남녀 섞어서의 성향 (아이디 -2)
            logger.info("Creating profile for all users (male and female)...")
            all_song_ids, all_scores = await self.fetch_gender_based_actions()  # gender=None으로 전체 데이터 조회
            all_profile = await self.create_user_profile(all_song_ids, all_scores)
            all_embedding = await self.embed_user_profile(all_profile)
            self.insert_or_update_user_profile(
                member_id=-2,  # 전체 남녀 아이디 -2
                profile_vector=all_embedding,
                profile_string=all_profile,
            )
        except Exception as e:
            logger.error(f"Failed to create gender profiles: {e}")
            raise

    async def run(self):
        try:
            self.create_user_profile_collection()
            recent_user_ids = await self.fetch_recent_member_ids()
            user_data = await self.fetch_user_actions_for_ids(recent_user_ids)

            for member_id, user_info in user_data.items():
                user_profile = await self.create_user_profile(user_info["song_ids"], user_info["scores"])
                user_embedding = await self.embed_user_profile(user_profile)
                self.insert_or_update_user_profile(member_id, user_embedding, user_profile)

            await self.create_gender_profiles()
        except Exception as e:
            logger.error(f"Error during run: {e}")
            raise