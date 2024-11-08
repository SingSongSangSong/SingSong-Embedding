import os
import logging
import aiomysql
import random
import grpc
import traceback
from typing import List
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from pymilvus import Collection, connections
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2 import FunctionCallingWithTypesResponse, SongInfo
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2_grpc import FunctionCallingWithTypesRecommendServicer
from service.functionCallingPrompts import PromptsForFunctionCalling, ExtractCommonTraitService
from service.langchainServiceAgentGrpc import LangChainServiceAgentGrpc

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAiFunctionOutput(BaseModel):
    song_info_id: List[str]
    reason: List[str]

class QueryType(BaseModel):
    query_type: str
    song_name: List[str]
    artist_name: List[str]
    octave : str
    vocal_range: str
    gender : str
    year : str
    genre : List[str]
    situation : List[str]
    country: str

class RefineQuery(BaseModel):
    refined_query: str

octave_info_list = [
    '1옥타브도', '1옥타브도#', '1옥타브레', '1옥타브레#', '1옥타브미', '1옥타브파', '1옥타브파#', '1옥타브솔', '1옥타브솔#', '1옥타브라', '1옥타브라#', '1옥타브시',
    '2옥타브도', '2옥타브도#', '2옥타브레', '2옥타브레#', '2옥타브미', '2옥타브파', '2옥타브파#', '2옥타브솔', '2옥타브솔#', '2옥타브라', '2옥타브라#', '2옥타브시',
    '3옥타브도', '3옥타브도#', '3옥타브레', '3옥타브레#', '3옥타브미', '3옥타브파', '3옥타브파#', '3옥타브솔', '3옥타브솔#', '3옥타브라', '3옥타브라#', '3옥타브시',
    '4옥타브도', '4옥타브도#', '4옥타브레', '4옥타브레#', '4옥타브미', '4옥타브파', '4옥타브파#', '4옥타브솔', '4옥타브솔#', '4옥타브라', '4옥타브라#', '4옥타브시',
    '5옥타브도', '5옥타브도#', '5옥타브레'
]
situation_list = ['classics', 'ssum', 'breakup', 'carol', 'finale', 'dance', 'duet', 'rainy', 'office', 'wedding', 'military']
genre_list = ["국악", "발라드", "록/메탈", "댄스", "성인가요/트로트", "포크/블루스", "키즈", "창작동요", "국내영화", "국내드라마", "랩/힙합", "R&B/Soul", "인디음악", "애니메이션/웹툰", "만화", "교과서동요", "국외영화", "POP", "클래식", "크로스오버", "J-POP", "CCM", "게임", "컨트리", "재즈", "보컬재즈", "포크", "블루스", "일렉트로니카", "월드뮤직", "애시드/퓨전/팝", "국내뮤지컬"]
country_list = ["대한민국", "미국", "일본", "영국", "스웨덴", "캐나다", "아일랜드", "노르웨이", "독일", "덴마크", "콜롬비아", "브라질", "러시아", "바베이도스", "오스트레일리아", "프랑스", "쿠바", "스페인", "뉴질랜드", "자마이카", "말레이시아", "네덜란드", "푸에르토리코 연방", "나이지리아", "가나", "아르헨티나", "벨기에", "중국", "폴란드", "타이 왕국", "타이완"]


class FunctionCallingWithTypesServiceGrpc(FunctionCallingWithTypesRecommendServicer):
    def __init__(self):
        try:
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.milvus_host = os.getenv("MILVUS_HOST", "milvus-standalone")
            self.collection_name = "final_song_embeddings"
            connections.connect(alias="FunctionCallingTypesGrpc", host=self.milvus_host, port="19530")
            self.collection = Collection(self.collection_name)
            self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
            self.openai = OpenAI(api_key=self.OPENAI_API_KEY)
            self.asyncOpenai = AsyncOpenAI(api_key=self.OPENAI_API_KEY)
            self.llm = ChatOpenAI(
                temperature=0.5,
                max_tokens=4096,
                streaming=True,
                callbacks=[StreamingStdOutCallbackHandler()],
                model_name='gpt-4o-mini',
                api_key=self.OPENAI_API_KEY
            )
            self.vectorstore = Milvus(
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
                connection_args={"host": self.milvus_host, "port": "19530"},
                text_field="song_name"
            )
            self.retriever = self.vectorstore.as_retriever()
            self.db_host = os.getenv('DB_HOST')
            self.db_user = os.getenv('DB_USER')
            self.db_password = os.getenv('DB_PASSWORD')
            self.db_database = os.getenv('DB_DATABASE')
            self.db_port = 3306
            self.langchainAgent = LangChainServiceAgentGrpc()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
    
    async def setup_db_config(self):
        try:
            # 비동기 MySQL 연결 설정
            pool = await aiomysql.create_pool(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                db=self.db_database,
                port=self.db_port,
                charset='utf8mb4',
                cursorclass=aiomysql.DictCursor,
                autocommit=False  # 자동 커밋 설정
            )
            logger.info("DB 연결 성공")
            return pool

        except aiomysql.MySQLError as e:
            logger.error(f"MySQL 연결 실패: {e}")
            raise
    
    async def determine_query_type(self, query: str):
        """
        Ask OpenAI to determine the type of query (specific song/artist, mood/genre, or specific feature).
        """
        try:
            messages = PromptsForFunctionCalling(query=query).prompt_for_decision

            response = await self.asyncOpenai.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=messages,
                response_format=QueryType,
            )

            # Extract and return query type
            parsed_result = response.choices[0].message.parsed
            query_type = parsed_result.query_type
            return query_type, parsed_result
        except Exception as e:
            logger.error(f"Failed to determine query type: {e}")
            return None, [], []

    async def handle_single_song_artist(self, input_song_name: list[str], input_artist_name: list[str]):
        """
        Handles queries to find similar songs based on a song and artist using Milvus DB and LCEL-style chain.
        """
        try:
            # Create the query
            query = ""

            try:
                if len(input_song_name) >= 1:
                    query += f"Title : {input_song_name[0]} "
                if len(input_artist_name) >= 1:
                    query += f"Artist : {input_artist_name[0]} "
            except Exception as e:
                logger.error(f"Failed to create query: {e}")
                return None, f"쿼리: {query} - 노래와 가수의 쿼리를 생성하는 데 실패했습니다."
            
            ## 노래 제목은 입력을 하였는데 가수 이름을 입력하지 않은 경우
            try:
                if len(input_song_name) >= 1 and len(input_artist_name) == 0:
                    # If only the song name is provided, search for similar songs based on the song name
                    results = await self.vectorstore.as_retriever(search_kwargs=dict(k=20)).ainvoke(query)
                    retrieved_data = [doc.metadata.get("song_info_id") for doc in results]
                    return retrieved_data, "오로지 노래 제목만 입력하셨습니다. 노래 제목과 가수 이름을 함께 입력해주세요. 노래 제목과 가수 이름을 함께 입력하면 더 정확한 결과를 제공할 수 있습니다."
                elif len(input_song_name) == 0 and len(input_artist_name) >= 1:
                    # If only the artist name is provided, search for similar songs based on the artist name
                    results = await self.vectorstore.as_retriever(search_kwargs=dict(k=20)).ainvoke(query)
                    retrieved_data = [doc.metadata.get("song_info_id") for doc in results]
                    return retrieved_data, "오로지 가수 이름만 입력하셨습니다. 노래 제목과 가수 이름을 함께 입력해주세요. 노래 제목과 가수 이름을 함께 입력하면 더 정확한 결과를 제공할 수 있습니다."
            except Exception as e:
                logger.error(f"Failed to create query: {e}")
                return None, "쿼리 생성에 실패했습니다."
            
            ## 정상적으로 노래 제목과 가수 이름을 입력한 경우
            try:
                # 노래 제목과 가수에 알맞는 노래 정보를 찾기 위해 Milvus DB에서 검색
                results = await self.vectorstore.as_retriever(search_kwargs=dict(k=1)).ainvoke(query)  # Example retriever setup
                # Print search results
                if not results:
                    print("No results found.")
                    return None, "노래에 대한 검색 결과가 없습니다."
            except Exception as e:
                logger.error(f"Failed to retrieve documents: {e}")
                return None, "노래에 대한 검색 결과를 가져오는 데 실패했습니다."
            
            ## 만약 노래 제목과 가수 이름을 입력하였을 때 결과가 존재하는 경우
            if results and len(results) > 0:
                # Combine the retrieved document descriptions into one text block
                tags = []

                db = await self.setup_db_config()
                async with db.acquire() as conn:
                    async with conn.cursor() as cursor:
                        query = f"""
                            SELECT
                                *
                            FROM song_info
                            WHERE song_info_id = %s;
                        """
                        await cursor.execute(query, (results[0].metadata.get("song_info_id"),))
                        songs = await cursor.fetchall()

                if len(songs) > 0:
                    songs = songs[0]

                # 정보들을 가져온다
                genre = songs["genre"]
                artist_type = songs["artist_type"]
                artist_name = songs["artist_name"]  # Assuming artist_name is part of the metadata
                year = songs["year"]
                country = songs["country"]
                lyrics_summary = songs["lyrics_summary"]
                artist_gender = songs["artist_gender"]  # Assuming this is part of the metadata
                octave = songs["octave"]  # Assuming this is part of the metadata

                # Extract Tag For Situation
                classics = songs["classics"]
                finale = songs["finale"]
                high = songs["high"]
                low = songs["low"]
                rnb = songs["rnb"]
                breakup = songs["breakup"]
                ballads = songs["ballads"]
                dance = songs["dance"]
                duet = songs["duet"]
                ssum = songs["ssum"]
                carol = songs["carol"]
                rainy = songs["rainy"]
                pop = songs["pop"]
                office = songs["office"]
                wedding = songs["wedding"]
                military = songs["military"]

                # Create the main content block, adding only if the value exists
                content = ""
                
                if genre:
                    content += f"Genre: {genre}\n"
                if year:
                    content += f"Year: {year}\n"
                if country:
                    content += f"Country: {country}\n"
                if artist_type:
                    content += f"Artist Type: {artist_type}\n"
                if artist_gender:
                    content += f"Artist Gender: {artist_gender}\n"
                if octave:
                    content += f"Octave: {octave}\n"
                if lyrics_summary:
                    content += f"Lyrics Summary: {lyrics_summary}\n"

                # Add tags based on metadata
                if classics: tags.append("모두가 인정한 명곡입니다.")
                if finale: tags.append("마무리 곡으로 자주 선택됩니다.")
                if low: tags.append("최고음역대가 매우 낮은 음역대의 곡입니다.")
                if high: tags.append("최고음역대가 매우 높은 것이 돋보이는 곡입니다.")
                if rnb: tags.append("R&B 요소를 포함하고 있습니다.")
                if ballads: tags.append("발라드 장르에 속한 곡입니다.")
                if breakup: tags.append("이별의 감정을 담고 있습니다.")
                if dance: tags.append("댄스 리듬이 가미된 곡입니다.")
                if duet: tags.append("친구와 듀엣으로 부르기 좋은 곡입니다.")
                if ssum: tags.append("썸 타는 상황에 어울리는 곡입니다.")
                if carol: tags.append("크리스마스 분위기를 담고 있습니다.")
                if rainy: tags.append("비 오는 날 듣기 좋은 곡입니다.")
                if pop: tags.append("팝송입니다.")
                if office: tags.append("사회생활할 때 아주 좋은 노래입니다. 나이가 있으신 분들이 알만한 노래로, 뽕짝의 느낌이 강하고 분위기를 살리기 위한 곡입니다.")
                if wedding: tags.append("결혼식에서 사용하기 좋은, 누군가를 축하할 때 적합한 곡입니다.")
                if military: tags.append("군대와 관련된 노래입니다.")

                # Combine tags into a "Situations" block
                if tags:
                    content += f"Situations:\n{' '.join(tags)}"

                # Search With Milvus DB for Similar Songs in Two Way - 1. Using Artist Name 2. Using Features
                async with db.acquire() as conn:
                    async with conn.cursor() as cursor:
                        query = """
                            SELECT song_info_id
                            FROM (
                                SELECT song_info_id
                                FROM song_info
                                WHERE artist_name = %s
                                ORDER BY melon_likes DESC
                                LIMIT 15
                            ) AS top_songs
                            ORDER BY RAND()
                            LIMIT 10
                        """
                        await cursor.execute(query, (artist_name,))
                        with_artist_info = await cursor.fetchall()
                
                with_feature_info = await self.vectorstore.asimilarity_search(content, k=10)

                #Combine the results from both searches
                song_info_ids_artist_info = [artist_info['song_info_id'] for artist_info in with_artist_info]
                song_info_ids_feature_info = [feature_info.metadata.get('song_info_id') for feature_info in with_feature_info]

                # Combine the results from both searches
                song_info_ids = song_info_ids_artist_info + song_info_ids_feature_info
                return song_info_ids, "유사한 노래를 찾았습니다."
            
            return None, "유사한 노래를 찾지 못했습니다."
        except Exception as e:
            logger.error(f"Error handling single song-artist query: {e}")
            return None, "유사한 노래를 찾지 못했습니다."

    async def handle_multiple_song_artist(self, input_song_name: list[str], input_artist_name: list[str]):
        """
        Handles queries with multiple song-artist pairs, finds common features and retrieves songs based on those.
        """
        try:
            if len(input_song_name) <= 1 and len(input_artist_name) <= 1:
                return None, "검색어가 부족합니다 - 노래 제목과 가수 이름을 둘 다 입력해주세요."

            totalSize = 20
            lengthOfSong = len(input_song_name)
            lengthOfArtist = len(input_artist_name)
            final_results = []

            # Step 1: 사용자가 노래 제목만 입력했는지 확인, 노래 제목만 입력한 경우 노래 제목과 관련된 노래를 찾는다.
            if lengthOfSong >= 2 and lengthOfArtist < lengthOfSong:
                lengthOfInput = lengthOfSong
                for song in input_song_name:
                    query = f"Title : {song} "
                    response = await self.vectorstore.asimilarity_search(query, totalSize//lengthOfInput)
                    if len(response) > 0:
                        final_results.extend([doc.metadata.get("song_info_id") for doc in response])
                
                return final_results, "오로지 노래 제목만 입력하셨습니다. 노래 제목과 가수 이름을 함께 입력해주세요. 노래 제목과 가수 이름을 함께 입력하면 더 정확한 결과를 제공할 수 있습니다."
            
            # Step 2: 사용자가 가수 이름만 입력했는지 확인, 가수 이름만 입력한 경우 가수 이름과 관련된 노래를 찾는다.
            elif lengthOfSong < lengthOfArtist and lengthOfArtist >= 2:
                lengthOfInput = len(input_artist_name)
                for artist in input_artist_name:
                    query = f"Artist : {artist} "
                    response = await self.vectorstore.asimilarity_search(query, totalSize//lengthOfInput)
                    if len(response) > 0:
                        final_results.extend([doc.metadata.get("song_info_id") for doc in response])
                
                return final_results, "오로지 가수 이름만 입력하셨습니다. 노래 제목과 가수 이름을 함께 입력해주세요. 노래 제목과 가수 이름을 함께 입력하면 더 정확한 결과를 제공할 수 있습니다."

            # Step 3: 사용자가 노래 제목과 가수를 모두 입력했는지 확인하고 노래 제목과 가수 이름을 함께 입력한 경우 노래 제목과 가수 이름을 함께 검색한다.
            if lengthOfSong >= 2 and lengthOfArtist >= 2 and lengthOfSong == lengthOfArtist:
                for i in range(lengthOfSong):            
                    query = f"Title: {input_song_name[i]} \n Aritst: {input_artist_name[i]}"
                    results = await self.vectorstore.asimilarity_search(query, 1)
                    # Combine the descriptions from each result
                    description, artist_name = results[0].metadata.get("description"), results[0].metadata.get("aritst_name")
                    final_results.append((description, artist_name))
                
                # Step 3: Combine the retrieved data into a single string
                combined_retrieved_data = "\n".join([description for description, _ in final_results])
                extractService = ExtractCommonTraitService(asyncOpenai=self.asyncOpenai)
                messages = await extractService.ExtractCommonTraitService(combined_retrieved_data)

                totalResult = []
                # 숫자 카운트를 위한 변수 추가
                artistResultNumber = totalSize//(lengthOfArtist*2)
                # Get Recommendation with Same Artist, and Features
                for _, artist in final_results:
                    response = await self.vectorstore.asimilarity_search(f"Artist : {artist}", artistResultNumber)
                    if len(response) > 0:
                        totalResult.extend([doc.metadata.get("song_info_id") for doc in response])

                # Extract and return query type
                feature_message = ""

                if messages.genre:
                    feature_message += "Genre: " + messages.genre + " "
                if messages.year:
                    feature_message += "Year: " + str(messages.year) + " "
                if messages.country:
                    feature_message += "Country: " + messages.country + " "
                if messages.artist_type:
                    feature_message += "Artist Type: " + messages.artist_type + " "
                if messages.artist_gender:
                    feature_message += "Artist Gender: " + messages.artist_gender + " "
                if messages.octave:
                    feature_message += "Octave: " + messages.octave + " "
                if messages.vocal_range:
                    feature_message += "Vocal Range: " + messages.vocal_range + " "
                if len(messages.situation) > 0:
                    feature_message += "Situation: " + " ".join(messages.situation) + " "
                if messages.lyrics:
                    feature_message += "Lyrics: " + messages.lyrics

                # 결과 출력
                print(feature_message.strip())  # 마지막에 불필요한 공백을 제거
                
                response_for_features = await self.vectorstore.asimilarity_search(feature_message, totalSize - artistResultNumber*lengthOfArtist)
                if len(response_for_features) > 0:
                    totalResult.extend([doc.metadata.get("song_info_id") for doc in response_for_features])
                
                return totalResult, "두 노래에 대해서 유사한 노래를 찾았습니다."
        except Exception as e:
            logger.error(f"Error handling multiple song-artist query: {e}")
            return None, "유사한 노래를 찾지 못했습니다."

    async def handle_octave_key(self, octave: str, gender: str):
        """
        Octave의 정보는 <MAX 3옥타브 레 or EQUAL 2옥타브 시 or MIN 1옥타브 레> 형식으로 입력됨
        """
        
        try:
            # 동적 필터링을 위한 기본 조건
            filters = ["octave IS NOT NULL"]
            params = []
            octave = octave.strip()

            # 성별 처리
            gender_filter = ""
            if gender and gender.lower().strip() in ["male", "female", "mixed"]:
                if gender.lower().strip() == "male":
                    gender_filter = f" AND artist_gender = '남성'"
                elif gender.lower().strip() == "female":
                    gender_filter = f" AND artist_gender = '여성'"
                elif gender.lower().strip() == "mixed":
                    gender_filter = f" AND artist_gender = '혼성'"

            # 데이터베이스 연결 및 쿼리 실행
            pool = await self.setup_db_config()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    query = f"""
                        SELECT
                            octave,
                            count(*) as count
                        FROM song_info
                        WHERE octave IS NOT NULL {gender_filter}  -- 성별 필터가 있으면 추가
                        GROUP BY octave
                        ORDER BY FIELD(
                            REPLACE(octave, ' ', ''),
                            '1옥타브도', '1옥타브도#', '1옥타브레', '1옥타브레#', '1옥타브미', '1옥타브파', '1옥타브파#', '1옥타브솔', '1옥타브솔#', '1옥타브라', '1옥타브라#', '1옥타브시',
                            '2옥타브도', '2옥타브도#', '2옥타브레', '2옥타브레#', '2옥타브미', '2옥타브파', '2옥타브파#', '2옥타브솔', '2옥타브솔#', '2옥타브라', '2옥타브라#', '2옥타브시',
                            '3옥타브도', '3옥타브도#', '3옥타브레', '3옥타브레#', '3옥타브미', '3옥타브파', '3옥타브파#', '3옥타브솔', '3옥타브솔#', '3옥타브라', '3옥타브라#', '3옥타브시',
                            '4옥타브도', '4옥타브도#', '4옥타브레', '4옥타브레#', '4옥타브미', '4옥타브파', '4옥타브파#', '4옥타브솔', '4옥타브솔#', '4옥타브라', '4옥타브라#', '4옥타브시',
                            '5옥타브도', '5옥타브도#', '5옥타브레'
                        );
                    """
                    await cursor.execute(query)
                    octave_data = await cursor.fetchall()

            # 옥타브별 노래 개수 카운트 및 결과 저장
            total_count = 0
            selected_octaves = []

            # MAX, MIN, EQUAL 처리 (기존 로직 그대로 유지)
            if "MAX" in octave:
                target_octave = octave.replace("MAX ", "").strip().replace(" ", "")
                if target_octave in octave_info_list:
                    # Reverse the octave_info_list to start from the highest octave
                    reversed_octave_info_list = octave_info_list[::-1]
                    index = reversed_octave_info_list.index(target_octave)
                    reverse_octave_data = octave_data[::-1]
                    # Start from the target octave and go downward
                    for data in reverse_octave_data:
                        octave_value = data['octave']
                        count = data['count']
                        if octave_value in reversed_octave_info_list[index:]:
                            selected_octaves.append(octave_value)
                            total_count += count
                        if total_count >= 20:
                            break
            elif "MIN" in octave:
                target_octave = octave.replace("MIN ", "").strip()
                if target_octave in octave_info_list:
                    index = octave_info_list.index(target_octave)
                    for data in octave_data:
                        octave_value = data['octave']
                        count = data['count']
                        if octave_value in octave_info_list[index:]:
                            selected_octaves.append(octave_value)
                            total_count += count
                        if total_count >= 20:
                            break
            else:
                logging.info("NONE OF MAX, MIN returning nothing")

            # 옥타브별 비중 설정 (가까운 옥타브일수록 더 많이 선택)
            total_selected_octaves = len(selected_octaves)
            remaining_needed = 20
            results = []
            logging.info(f"Total selected octaves: {selected_octaves}")

            # 가중치 설정: 가까운 옥타브에 높은 가중치 (반비례 가중치 방식)
            weights = [(1 / (i + 1)) for i in range(total_selected_octaves)]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]  # 가중치 합을 1로 맞춤
            logging.info(f"Selected octaves: {selected_octaves}, Weights: {weights}")

            for i, octave in enumerate(selected_octaves):
                async with pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        # 각 옥타브에서 가중치에 따라 곡 수를 결정 (최소 1곡은 보장)
                        proportion_count = int(weights[i] * remaining_needed)
                        if proportion_count == 0:
                            proportion_count = 1
                        
                        # 해당 옥타브에서 노래를 melon_likes 기준으로 가져옴
                        await cursor.execute(f"""
                            SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                            FROM song_info
                            WHERE octave = %s and is_mr = 0 and is_live = 0 {gender_filter}
                            ORDER BY melon_likes DESC
                            LIMIT {proportion_count * 2} -- 랜덤성을 위해 2배수로 가져옴
                        """, (octave,))
                        songs = await cursor.fetchall()

                        # melon_likes에서 일부는 랜덤으로 선택
                        random.shuffle(songs)
                        selected_songs = songs[:proportion_count]
                        results.extend([SongInfo(songNumber=song['song_number'], songName=song['song_name'], artistName=song['artist_name'], songInfoId=song['song_info_id'], album=song['album'], isMr=song['is_mr'], isLive=song['is_live'], melonSongId=song['melon_song_id'], lyricsYoutubeLink=song['lyrics_video_link'], tjYoutubeLink=song['tj_youtube_link']) for song in selected_songs])

                        remaining_needed -= len(selected_songs)
                        if remaining_needed <= 0:
                            break

            # 결과 반환 (선택된 song_info_id 리스트)
            return results, ", ".join(selected_octaves) + " 옥타브의 노래를 추천합니다."

        except Exception as e:
            logger.error(f"Failed to handle octave key: {e}")
            return None, "옥타브 정보를 처리하는 중 오류가 발생했습니다."

    async def handle_octave_song_artist_key(self, song_name: List[str], artist_name: List[str], octave: str):
        if(len(song_name) == 0 or len(artist_name) == 0 or octave == None):
            return None, "노래 제목, 가수 이름, 옥타브 정보를 모두 입력해주세요."
        try:
            # Create the query
            query = ""

            try:
                if len(song_name) >= 1:
                    query += f"Title : {song_name[0]} "
                if len(artist_name) >= 1:
                    query += f"Artist : {artist_name[0]} "
                print("query: " + query)
            except Exception as e:
                logger.error(f"Failed to create query: {e}")
                return None, "쿼리: {query} - 노래와 가수의 쿼리를 생성하는 데 실패했습니다."
            
            try:
                # 노래 제목과 가수에 알맞는 노래 정보를 찾기 위해 Milvus DB에서 검색
                results = await self.vectorstore.as_retriever(search_kwargs=dict(k=1)).ainvoke(query)  # Example retriever setup
                # Print search results
                if not results:
                    print("No results found.")
                    return None, "노래에 대한 검색 결과가 없습니다."
            except Exception as e:
                logger.error(f"Failed to retrieve documents: {e}")
                return None, "노래에 대한 검색 결과를 가져오는 데 실패했습니다."
            
            ## 만약 노래 제목과 가수 이름을 입력하였을 때 결과가 존재하는 경우
            if results and len(results) > 0:
                # Combine the retrieved document descriptions into one text block
                tags = []

                ## 옥타브 정보들을 가져온다
                octave = results[0].metadata.get("octave")
                print(f"Octave: {octave}")

                ## 옥타브 정보가 존재하는 경우
                if octave:
                    # 옥타브 정보를 통해 노래 추천
                    song_info_ids = await self.handle_octave_key("MAX " + octave.strip().replace(" ", ""), None)
                    return song_info_ids, "유사한 노래를 찾았습니다."
                else:
                    return None, "옥타브 정보가 없습니다."
                
        except Exception as e:
            logger.error(f"Error handling single song-artist query: {e}")
            return None, "유사한 노래를 찾지 못했습니다."
        
    async def handle_hit_songs(self, artist_name: List[str]):
        try:
            # artist_name이 주어졌을 경우 처리
            found_artist_name = None
            if len(artist_name) != 0:
                query = f"Artist : {artist_name[0]}"
                response = await self.vectorstore.asimilarity_search(query, 1)
                if len(response) > 0:
                    found_artist_name = response[0].metadata.get("artist_name")
            
            # 데이터베이스 연결 및 쿼리 실행
            pool = await self.setup_db_config()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    # melon_likes 기준으로 상위 1500곡을 가져옴
                    if found_artist_name:
                        # artist_name이 존재하는 경우
                        await cursor.execute("""
                            SELECT song_info_id, song_number, song_name, artist_name, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                            FROM (
                                SELECT song_info_id, song_number, song_name, artist_name, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                                FROM song_info
                                WHERE artist_name = %s
                                ORDER BY melon_likes DESC
                                LIMIT 1500
                            ) AS top_songs
                            ORDER BY RAND()
                            LIMIT 20;
                        """, (found_artist_name,))
                    else:
                        # artist_name이 존재하지 않는 경우
                        await cursor.execute("""
                            SELECT song_info_id, song_number, song_name, artist_name, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                            FROM (
                                SELECT song_info_id, song_number, song_name, artist_name, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                                FROM song_info
                                ORDER BY melon_likes DESC
                                LIMIT 1500
                            ) AS top_songs
                            ORDER BY RAND()
                            LIMIT 20;
                        """)
                    
                    songs = await cursor.fetchall()

            # 결과 반환 (랜덤으로 20곡 선택)
            song_info_ids = [SongInfo(songNumber=song['song_number'], songName=song['song_name'], artistName=song['artist_name'], songInfoId=song['song_info_id'], album=song['album'], isMr=song['is_mr'], isLive=song['is_live'], melonSongId=song['melon_song_id'], lyricsYoutubeLink=song['lyrics_video_link'], tjYoutubeLink=song['tj_youtube_link']) for song in songs]
            return song_info_ids, "인기 있는 노래를 추천합니다."

        except Exception as e:
            print(f"Error: {e}")
            return [], "인기 있는 노래를 찾지 못했습니다."

    async def handle_vocal_range(self, vocal_range, gender, genre):
        """
        Handles vocal range queries (high or low) with an optional gender filter (남성, 여성, 혼성).
        """
        if vocal_range is None:
            return None, "음역대 정보를 입력해주세요."

        try:
            db = await self.setup_db_config()
            async with db.acquire() as conn:
                async with conn.cursor() as cursor:
                    # Prepare the gender filter if gender is provided and valid
                    gender_filter = ""
                    # 성별 처리
                    if gender and gender.lower().strip() in ["male", "female", "mixed"]:
                        if gender.lower().strip() == "male":
                            gender_filter = " AND artist_gender = '남성'"
                        elif gender.lower().strip() == "female":
                            gender_filter = " AND artist_gender = '여성'"
                        elif gender.lower().strip() == "mixed":
                            gender_filter = " AND artist_gender = '혼성'" 

                    # 유효한 장르만 필터링
                    valid_genres = [a_genre.strip() for a_genre in genre if a_genre.strip() and a_genre.strip() in genre_list]

                    # 장르 처리
                    genre_filter = ""
                    if valid_genres:
                        # 각각의 장르에 대해 'genre = 장르' 조건을 추가하고 OR로 묶음
                        genre_conditions = " OR ".join(f"genre = '{g}'" for g in valid_genres)
                        genre_filter = f" AND ({genre_conditions})"

                    print(gender_filter + genre_filter) 

                    # Handle high vocal range with combined filters
                    if vocal_range == "high":
                        await cursor.execute(f"""
                            SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                            FROM (
                                SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                                FROM song_info
                                WHERE high = 1 {gender_filter} {genre_filter} AND is_mr = 0 AND is_live = 0
                                ORDER BY melon_likes DESC
                                LIMIT 500
                            ) AS top_songs
                            ORDER BY RAND()
                            LIMIT 20;
                        """)
                    
                    # Handle low vocal range
                    elif vocal_range == "low":
                        await cursor.execute(f"""
                            SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                            FROM (
                                SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                                FROM song_info
                                WHERE low = 1 {gender_filter} {genre_filter} and is_mr = 0 and is_live = 0
                                ORDER BY melon_likes DESC
                                LIMIT 500
                            ) AS top_songs
                            ORDER BY RAND()
                            LIMIT 20;
                        """)

                    songs = await cursor.fetchall()
            return [SongInfo(songNumber=song['song_number'], songName=song['song_name'], artistName=song['artist_name'], songInfoId=song['song_info_id'], album=song['album'], isMr=song['is_mr'], isLive=song['is_live'], melonSongId=song['melon_song_id'], lyricsYoutubeLink=song['lyrics_video_link'], tjYoutubeLink=song['tj_youtube_link']) for song in songs], f"{vocal_range} 음역대의 노래를 추천합니다."
        
        except Exception as e:
            logger.error(f"Failed to handle vocal range: {e}")
            return None

    async def handle_situation(self, situation: List[str], gender: str):
        """
        Handles situation-based queries. Filters the situation list to match valid situations,
        and queries the database for each valid situation with a dynamic LIMIT based on the number of situations.
        The last situation will get the remaining songs to make up a total of 20.
        """
        if len(situation) == 0 and gender is None:
            return None, "상황 정보를 입력해주세요."

        try:
            return_situation_list = []
            # 유효한 situation들만 필터링
            valid_situations = [s.lower().strip() for s in situation if s.lower().strip() in situation_list]

            if len(valid_situations) == 0:
                return None, "Situation에 해당하지 않습니다"  # 유효한 상황이 없으면 None 반환
            
            # Prepare the gender filter if gender is provided and valid
            gender_filter = ""
            # 성별 처리
            if gender and gender.lower().strip() in ["male", "female", "mixed"]:
                if gender.lower().strip() == "male":
                    gender_filter = f" AND artist_gender = '남성'"
                elif gender.lower().strip() == "female":
                    gender_filter = f" AND artist_gender = '여성'"
                elif gender.lower().strip() == "mixed":
                    gender_filter = f" AND artist_gender = '혼성'"    


            # 각 상황에 대해 가져올 노래 수 계산 (20개를 상황의 개수로 나누어 제한)
            limit_per_situation = 20 // len(valid_situations)
            remaining_songs = 20  # 남은 노래 수 초기화

            # 데이터베이스 연결
            db = await self.setup_db_config()

            # 각 유효한 상황에 대해 노래를 검색
            async with db.acquire() as conn:
                async with conn.cursor() as cursor:
                    for i, s in enumerate(valid_situations):
                        if i == len(valid_situations) - 1:
                            # 마지막 상황에서는 남은 노래 수만큼 LIMIT 설정
                            limit = remaining_songs
                        else:
                            limit = limit_per_situation
                            remaining_songs -= limit_per_situation  # 남은 노래 수 차감

                        await cursor.execute(f"""
                            SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                            FROM (
                                SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                                FROM song_info
                                WHERE {s} = 1 and is_mr = 0 and is_live = 0{gender_filter}
                                ORDER BY melon_likes DESC
                                LIMIT 500
                            ) AS top_songs
                            ORDER BY RAND()
                            LIMIT {limit};
                        """)
                        songs = await cursor.fetchall()

                        # 결과를 합산하여 리스트에 추가
                        return_situation_list.extend([SongInfo(songNumber=song['song_number'], songName=song['song_name'], artistName=song['artist_name'], songInfoId=song['song_info_id'], album=song['album'], isMr=song['is_mr'], isLive=song['is_live'], melonSongId=song['melon_song_id'], lyricsYoutubeLink=song['lyrics_video_link'], tjYoutubeLink=song['tj_youtube_link']) for song in songs])

            # 모든 상황에 대해 검색한 노래 ID 리스트 반환
            return return_situation_list, ", ".join(valid_situations) + " 상황에 어울리는 노래를 추천합니다."

        except Exception as e:
            logger.error(f"Failed to handle situation: {e}")
            return None, "상황 정보를 처리하는 중 오류가 발생했습니다."

    async def handle_year_gender_genre(self, year: str, gender: str, genre: List[str], country: str):
        """
        Handles year, gender, and genre-based queries.
        If a year range or specific year is provided, it retrieves songs from that period.
        It retrieves the top 500 songs based on melon_likes, and then selects 20 songs randomly.
        """
        if year is None and gender is None and genre is None:
            return None, "연도, 성별, 장르 정보를 입력해주세요."

        try:
            db = await self.setup_db_config()

            # 기본 SQL 쿼리 시작 부분
            base_query = """
                SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                FROM (
                    SELECT song_number, song_name, artist_name, song_info_id, album, is_mr, is_live, melon_song_id, lyrics_video_link, tj_youtube_link
                    FROM song_info
                    WHERE 1=1 and is_mr = 0 and is_live = 0
            """

            # SQL 조건을 저장할 리스트
            conditions = []

            # 연도 처리
            if year:
                if "&&" in year:
                    # 연도 범위 처리 (예: "year >= 2010 && year <= 2019")
                    start_year, end_year = year.split("&&")
                    if ">=" in start_year or "<=" in end_year:
                        conditions.append(f"{start_year.strip()} AND {end_year.strip()}")
                elif "==" in year:
                    # 특정 연도 처리 (예: "2013")
                    year_query = year.replace("==", "=").strip()
                    conditions.append(f"year = {year_query}")

            # 성별 처리
            if gender and gender.lower().strip() in ["male", "female", "mixed"]:
                if gender.lower().strip() == "male":
                    conditions.append(f"artist_gender = '남성'")
                elif gender.lower().strip() == "female":
                    conditions.append(f"artist_gender = '여성'")
                elif gender.lower().strip() == "mixed":
                    conditions.append(f"artist_gender = '혼성'")        

            # 유효한 장르만 필터링
            valid_genres = [a_genre.strip() for a_genre in genre if a_genre.strip() and a_genre.strip() in genre_list]

            # 장르 처리
            if valid_genres:
                # 각각의 장르에 대해 'genre = 장르' 조건을 추가하고 AND로 묶음
                genre_conditions = " AND ".join(f"genre = '{g}'" for g in valid_genres)
                conditions.append(f"({genre_conditions})")

            if country:
                if country in country_list:
                    conditions.append(f"country = '{country}'")

            # 조건들을 AND로 연결
            if conditions:
                query = base_query + " AND " + " AND ".join(conditions)
            else:
                query = base_query

            # 서브쿼리로 상위 500곡을 가져온 후 무작위로 20곡 선택
            query += """
                    ORDER BY melon_likes DESC
                    LIMIT 500
                ) AS top_songs
                ORDER BY RAND()
                LIMIT 20;
            """

            async with db.acquire() as conn:
                async with conn.cursor() as cursor:
                    # SQL 쿼리 실행
                    await cursor.execute(query)
                    songs = await cursor.fetchall()

            # 결과 반환
            return [SongInfo(songNumber=song['song_number'], songName=song['song_name'], artistName=song['artist_name'], songInfoId=song['song_info_id'], album=song['album'], isMr=song['is_mr'], isLive=song['is_live'], melonSongId=song['melon_song_id'], lyricsYoutubeLink=song['lyrics_video_link'], tjYoutubeLink=song['tj_youtube_link']) for song in songs], "조건에 맞는 노래를 추천합니다."

        except Exception as e:
            logger.error(f"Failed to handle year, gender, or genre: {e}")
            return None, "연도, 성별, 장르 정보를 처리하는 중 오류가 발생했습니다."
    
    async def get_song_infos(self, song_info_ids: List[str]):
        """
        Retrieves song information based on song_info_ids, sorted by melon_likes (default to 0 if missing).
        """
        try:
            db = await self.setup_db_config()
            song_infos = []

            async with db.acquire() as conn:
                async with conn.cursor() as cursor:
                    # song_info_ids를 한 번의 쿼리로 처리하고 melon_likes를 기준으로 정렬
                    query = """
                        SELECT 
                            song_number, 
                            song_name, 
                            artist_name, 
                            song_info_id, 
                            album, 
                            is_mr, 
                            is_live, 
                            melon_song_id,
                            COALESCE(melon_likes, 0) AS melon_likes,
                            lyrics_video_link,
                            tj_youtube_link
                        FROM song_info
                        WHERE song_info_id IN %s
                        ORDER BY melon_likes DESC
                    """
                    # 튜플로 song_info_ids를 넘겨줍니다.
                    await cursor.execute(query, (tuple(song_info_ids),))
                    song_infos = await cursor.fetchall()
                    result = [SongInfo(songNumber=song['song_number'], songName=song['song_name'], artistName=song['artist_name'], songInfoId=song['song_info_id'], album=song['album'], isMr=song['is_mr'], isLive=song['is_live'], melonSongId=song['melon_song_id'], lyricsYoutubeLink=song['lyrics_video_link'], tjYoutubeLink=song['tj_youtube_link']) for song in song_infos]

            return result

        except Exception as e:
            logger.error(f"Failed to get song info: {e}")
            return None

    async def handle_features_with_agent(self, query:str, features: QueryType):
        """
        Handles queries with multiple features (genre, year, country, artist type, artist gender, etc.).
        """
        try:
            # Step 1: Extract the features from the query
            song_name = features.song_name
            artist_name = features.artist_name
            genre = features.genre
            year = features.year
            country = features.country
            octave = features.octave
            vocal_range = features.vocal_range
            situation = features.situation
            country = features.country
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return None, "특징을 추출하는 중 오류가 발생했습니다."
    
        try:
            # Step 2: Prepare the query for the LangChain service
            query = f"User Query: {query} \n Information extracted from query: \n Title : {song_name} Artist : {artist_name} Genre : {genre} Year : {year} Country : {country} Octave : {octave} VocalRange : {vocal_range} Situation : {situation}"
            song_info_ids, reason_list = await self.langchainAgent.run(query)
            return song_info_ids, reason_list
        except Exception as e:
            logger.error(f"Failed to handle features with agent: {e}")
            return None, "특징을 처리하는 중 오류가 발생했습니다."

    async def run(self, query: str):
        """
        Main method to handle various types of user inputs and return recommendations.
        """
        try:
            songInfos = []
            message = "유사한 노래를 찾지 못했습니다."
            try:
                logging.info(f"Received query: {query}")
                query_type, Results = await self.determine_query_type(query)
                logging.info(f"Results: {Results}")
                if query_type == "single_song_artist":
                    songInfos, message = await self.handle_single_song_artist(Results.song_name, Results.artist_name)
                    if len(songInfos) > 0:
                        songInfos = await self.get_song_infos(songInfos)
                    logging.info(f"song_info_ids from single query: {songInfos}")
                elif query_type == "multiple_song_artist":
                    songInfos, message = await self.handle_multiple_song_artist(Results.song_name, Results.artist_name)
                    if len(songInfos) > 0:
                        songInfos = await self.get_song_infos(songInfos)
                    logging.info(f"result_song_ids from multi query: {songInfos}")
                elif query_type == "octave_key":
                    songInfos, message = await self.handle_octave_key(Results.octave, Results.gender)
                    logging.info(f"song_info_ids from octave_key query: {songInfos}")
                elif query_type == "song_artist_octave":
                    songInfos, message = await self.handle_octave_song_artist_key(Results.song_name, Results.artist_name, Results.octave)
                    logging.info(f"song_info_ids from song_artist_octave query: {songInfos}")
                elif query_type == "hit_songs":
                    songInfos, message = await self.handle_hit_songs(Results.artist_name)
                    logging.info(f"song_info_ids from hit_songs query: {songInfos}")
                elif query_type == "vocal_range":
                    songInfos, message = await self.handle_vocal_range(Results.vocal_range, Results.gender, Results.genre)
                    logging.info(f"song_info_ids from vocal_range query: {songInfos}")
                elif query_type == "situation":
                    songInfos, message = await self.handle_situation(Results.situation, Results.gender)
                    logging.info(f"song_info_ids from situation query: {songInfos}")
                elif query_type == "year_gender_genre":
                    songInfos, message = await self.handle_year_gender_genre(Results.year, Results.gender, Results.genre, Results.country)
                    logging.info(f"song_info_ids from year gender genre query: {songInfos}")
                else:
                    logging.info("No valid query type found")
            
                if songInfos is None or len(songInfos) == 0 or query_type == "None":
                    songInfos, message = await self.handle_features_with_agent(query, Results)
                    if songInfos is None:
                        return {"songInfos": [], "message": "유사한 노래를 찾지 못했습니다."}
                    elif len(songInfos) == 0:
                        return {"songInfos": [], "message": "유사한 노래를 찾지 못했습니다."}
                    elif len(songInfos) > 0:
                        songInfos = await self.get_song_infos(songInfos)
                        message = "싱송이의 AI 전체 검색 결과입니다."
                    logging.info(f"song_info_ids from features query: {songInfos}")
                        
            except Exception as e:
                logger.error(f"Failed to determine query type: {e}")
                return {"songInfos": [], "message": "유사한 노래를 찾지 못했습니다."}

            return {
                "songInfos": songInfos,
                "message": message
            }

        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            return None
        
    async def GetFunctionCallingWithTypesRecommendation(self, request, context):
        try:
            # 유사한 노래 검색 (song_info_id 리스트가 반환됨)
            search_results = await self.run(request.command)  # 유사한 노래 검색 (song_info_id 리스트)
            if search_results is None:
                logger.error("Search results are None.")
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details("No valid search results")
            return FunctionCallingWithTypesResponse(songInfos = search_results["songInfos"], message=search_results["message"])
                
        except Exception as e:
            logger.error(f"Error during GetFunctionCallingWithTypesRecommendation: {e}")
            logger.error(traceback.format_exc())  # 전체 스택 트레이스를 출력합니다.
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return FunctionCallingWithTypesResponse(songInfos=[], message="No valid search results")
