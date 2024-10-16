import os
import logging
from typing import List
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from pymilvus import Collection, connections, AnnSearchRequest, RRFRanker, WeightedRanker
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2 import FunctionCallingWithTypesRequest, FunctionCallingWithTypesResponse
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2_grpc import FunctionCallingWithTypesRecommendServicer
import grpc
import traceback
import logging
import os
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_milvus import Milvus

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
    genre : str
    situation : str

class RefineQuery(BaseModel):
    refined_query: str

class FunctionCallingWithTypesServiceGrpc(FunctionCallingWithTypesRecommendServicer):
    def __init__(self):
        try:
            self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            self.milvus_host = os.getenv("MILVUS_HOST", "milvus-standalone")
            self.collection_name = "singsongsangsong_22286"
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
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
    
    async def determine_query_type(self, query: str):
        """
        Ask OpenAI to determine the type of query (specific song/artist, mood/genre, or specific feature).
        """
        try:
            messages = [
                {"role": "system", "content": """
                You are an assistant that categorizes user inputs into specific types of song recommendations and extracts important information.

                Follow these guidelines for classifying the query into one of the following nine categories and extracting relevant information, ensuring all relevant fields are extracted even if the information is not provided.

                1. **Single Song/Artist Query:**
                    - If the user input contains either a specific song title, an artist name, or both, asking for similar songs.
                    - If both a song title and artist name are mentioned, treat it as a full song-artist query.
                    - If only a song title or artist name is mentioned, return recommendations based on the available information.
                    - **Extract**: song name, artist name (If not provided, return an empty list `[]` or None).
                    - **For example**: 
                        - "버즈의 가시나 같은 노래 추천해줘".
                        - "추억은 만남보다 이별에 남아 같은 노래 찾아줘".
                        - "엠씨더맥스 노래 추천해줘".
                    - Output query type as 'single_song_artist'.

                2. **Multiple Song-Artist Pairs Query:**
                    - If the user provides multiple song-artist pairs or only song titles or artist names and is asking for recommendations based on the common features between those pairs.
                    - **Extract**: all song names, all artist names (If not provided, return an empty list `[]` or None).
                    - **For example**:
                        - "Recommend me songs like 'Bang Bang Bang' by BigBang and 'Thorn' by Buzz".
                        - "마이클 잭슨의 Thriller랑 퀸의 Bohemian Rhapsody 같은 곡 추천해줘".
                        - "아이유와 태연의 노래 추천해줘".
                        - "가시랑 거짓말 같은 노래 추천해줘".
                    - Output query type as 'multiple_song_artist'.

                3. **Octave/Key-based Query:**
                    - If the user mentions specific octaves, vocal ranges (high or low), or difficulty related to singing (e.g., easy or hard songs).
                    - The query might involve asking for songs with specific vocal demands, such as high-pitched, low-pitched, or songs in a particular octave.
                    - **Extract**: octave information, vocal range (high/low) (If not provided, return `None`).
                    - **For example**:
                        - "노래방 인기차트 중에서 남자가 부르기 쉬운, 음역대 낮은 노래들".
                        - "최고음 2옥타브 시 노래 추천해줘".
                        - "고음 어려운 곡 추천해줘".
                        - "쉬운 발라드 추천해줘".
                    - Output query type as 'octave_key'.

                4. **Octave with Song/Artist Query:**
                    - If the user provides both a specific song and mentions octaves or key changes.
                    - **Extract**: song name, artist name, octave information (If not provided, return an empty list `[]` or `None`).
                    - **For example**:
                        - "김연우의 '내가 너의 곁에 잠시 살았다는걸' 정도의 음역대의 노래 추천해줘".
                        - "Lower the key for 'Imagine' by John Lennon".
                    - Output query type as 'song_artist_octave'.

                5. **Vocal Range (High/Low) Query:**
                    - If the user asks for songs based on vocal range, such as high-pitched or low-pitched songs.
                    - **Extract**: vocal range (high/low) (If not provided, return `None`).
                    - **For example**:
                        - "고음 자신 있는데, 고음 폭발하는 곡 추천해줘".
                        - "남자 저음발라드 추천해줘".
                        - "음역대가 낮은 남자 노래 추천해줘".
                    - Output query type as 'vocal_range'.

                6. **Situation-Based Query (Breakup, Christmas, Ssum):**
                    - If the user is asking for songs based on specific situations or occasions.
                    - **Extract**: situation context (e.g., breakup, Christmas, ssum, etc.), gender if applicable (If not provided, return `None`).
                    - **For example**:
                        - "감성적인 이별 노래 추천해줘".
                        - "여자친구 앞에서 부를만한 노래".
                        - "오늘 헤어졌는데 부를만한 노래 추천좀..".
                        - "썸탈때 부를만한 남자노래 추천해줘".
                    - **Situation Keywords**: 반드시 다음 목록의 상황 키워드로 반환되어야 합니다:
                        - 그시절 띵곡 -> classics
                        - 썸 -> ssum
                        - 이별/헤어졌을때 -> breakup
                        - 크리스마스/캐롤/눈 -> carol
                        - 마지막곡 -> finale
                        - 신나는/춤/흥 -> dance
                        - 듀엣 -> duet
                        - 비/꿀꿀할때 -> rainy
                        - 회사생활 -> office
                        - 축하 -> wedding
                        - 군대/입대 -> military
                    - Output query type as 'situation'.

                7. **Year/Gender/Genre-based/Solo-Group Based Query:**
                    - If the user asks for songs based on a specific year, gender, genre, or whether it's suitable for solo or group singing.
                    - If gender exists in the query You have to decide which gender that the user wants to get recommendations for.
                    - **Extract**: year, genre, gender (female/male/mixed), performance type (solo/group) (If not provided, return `None` or an empty list).
                    - When extracting **genre**, return the most appropriate match from the following list of genres from the database:
                        - 국악
                        - 발라드
                        - 록/메탈
                        - 댄스
                        - 성인가요/트로트
                        - 포크/블루스
                        - 키즈
                        - 창작동요
                        - 국내영화
                        - 국내드라마
                        - 랩/힙합
                        - R&B/Soul
                        - 인디음악
                        - 애니메이션/웹툰
                        - 만화
                        - 교과서동요
                        - 국외영화
                        - POP
                        - 클래식
                        - 크로스오버
                        - J-POP
                        - CCM
                        - 게임
                        - 컨트리
                        - 재즈
                        - 보컬재즈
                        - 포크
                        - 블루스
                        - 일렉트로니카
                        - 월드뮤직
                        - 애시드/퓨전/팝
                        - 국내뮤지컬
                    - **For example**:
                        - "2010년도 쯤에 신나는 노래 추천해줘".
                        - "발라드 2024".
                        - "혼자 노래방 갈 때 부르기 좋은 노래 있을까?".
                        - "여자 가수들의 힙합곡 추천".
                    - Output query type as 'year_gender_genre'.

                It's important to ensure that the query fits into only one of these nine categories. If the user input is unclear, make your best effort to infer the most likely category.
                Always extract all the relevant fields, even if the information is missing, by returning `None` or an empty list where applicable.

                Format:
                - Query Type: <single_song_artist/multiple_song_artist/octave_key/song_artist_octave/hit_songs/vocal_range/situation/year_gender_genre>
                - Song Name: [<song_name1>, <song_name2>, ...] (If applicable, otherwise `[]`)
                - Artist Name: [<artist_name1>, <artist_name2>, ...] (If applicable, otherwise `[]`)
                - Octave: [<octave_info>] (If applicable, otherwise `None`)
                - Vocal Range: [<vocal_range>] (high, low, or `None`)
                - Gender: [<gender_info>] (female, male, mixed, or `None`)
                - Year: [<year_info>] (If year range, return `year >= start && year <= end`, otherwise `None`)
                - Genre: [<genre_info>] (If applicable, otherwise `None`)
                - Situation: [<situation_info>] (If applicable, otherwise `None`)
                """},
                {"role": "user", "content": query}
            ]

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

    async def handle_single_song_artist(self, song_name: list[str], artist_name: list[str]):
        """
        Handles queries to find similar songs based on a song and artist using Milvus DB and LCEL-style chain.
        """
        try:
            # Create the query
            query = ""

            print("song_name type: " + str(type(song_name)))
            print("artist_name type: " + str(type(artist_name)))
            print("song_name: " + str(song_name))
            print("artist_name: " + str(artist_name))

            try:
                if len(song_name) >= 1:
                    query += f"Title : {song_name[0]}"
                if len(artist_name) >= 1:
                    query += f"Artist : {artist_name[0]}"
                print("query: " + query)
            except Exception as e:
                logger.error(f"Failed to create query: {e}")
                return None

            try:
                # Retrieve relevant documents from Milvus (assuming `retriever` is set up for Milvus)
                results = await self.retriever.ainvoke(query, )  # Example retriever setup
                # Print search results
                if results:
                    print(f"Search Results: {results}")
                else:
                    print("No results found.")
            except Exception as e:
                logger.error(f"Failed to retrieve documents: {e}")
                return None
            try:
                result2 = await self.vectorstore.asimilarity_search(query, k=1)
                if result2:
                    print(f"Search Results: {result2}")
                else:
                    print("No results found.")
            except Exception as e:
                logger.error(f"Failed to retrieve documents: {e}")
                return None
            
            # # Combine the retrieved document descriptions into one text block
            # retrieved_data = "\n".join([doc.metadata.get("description") for doc in results])

            # # Create the prompt using the retrieved data
            # prompt_template = PromptTemplate.from_template(
            #     """
            #     You are a music recommendation assistant. The user is asking for songs similar to the following query: "{query}". Below are descriptions of songs that were retrieved from a song database based on this query:
                
            #     {retrieved_data}
                
            #     Your task is to refine the user's query by analyzing the common characteristics of these songs, such as their genre, tempo, mood, vocal style, instrumentation, and era. Ensure that the recommendations are not limited to songs by the same artist.

            #     **Make sure that all recommended songs are from the 2000s, 2010s. and 2020s ** Specifically, analyze shared attributes such as energy level (high-energy or calm), instruments used (e.g., guitar, synthesizer, piano), vocal type (e.g., male or female vocals, solo or group), production style (e.g., acoustic, electronic), and mood (e.g., upbeat, melancholic, nostalgic).

            #     Then, provide a refined query suggesting songs with similar overall characteristics, even if they are from different artists or slightly different genres. **However, ensure the songs are from the 2000s or later.**

            #     **Important**: You must output the refined query in **one single, well-structured sentence**. The format of the output must exactly follow the example below:

            #     - **Refined Query**: Find songs from the 2010s or later that are <refined characteristics>, featuring <key shared features>, and explore <themes/moods>.

            #     Here is an example of the expected format:

            #     - Refined Query: Find songs from the 2010s or later that are in the ballad genre, featuring male solo vocals with a calm and nostalgic mood, moderate tempo, and soft instrumental textures, exploring themes of love and breakups.
            #     """
            # )

            # # Format the template with actual retrieved data
            # prompt = prompt_template.format(query=query, retrieved_data=retrieved_data)

            # logging.info(f"Prompt template: {prompt}")
            # # Now, pass the formatted prompt (as a string) to the LLM
            # response = await self.asyncOpenai.beta.chat.completions.parse(
            #     model="gpt-4o-mini",
            #     messages=[{"role": "system", "content": prompt}],
            #     response_format=RefineQuery,
            # )
            # parsed_result = response.choices[0].message.parsed
                    
            # Parse the response and return the recommendations
            return None
        except Exception as e:
            logger.error(f"Error handling single song-artist query: {e}")
            return None

    async def handle_multiple_song_artist(self, songs: list):
        """
        Handles queries with multiple song-artist pairs, finds common features and retrieves songs based on those.
        """
        try:
            # Step 1: Build the combined query for all the song-artist pairs
            query_for_langchain = "Find songs similar to the following songs:"
            for song in songs:
                query_for_langchain += f" '{song['song_name']}' by {song['artist_name']}."

            # Step 2: Retrieve relevant documents for each song
            documents = []
            for song in songs:
                query = f"Find songs similar to {song['song_name']} by {song['artist_name']}"
                results = await self.get_relevant_documents(query, 1)
                # Combine the descriptions from each result
                retrieved_data = "\n".join([doc.metadata.get("description") for doc in results])
                documents.append(retrieved_data)

            # Step 3: Combine the retrieved data into a single string
            combined_retrieved_data = "\n".join(documents)

            # Step 4: Create the detailed prompt for multiple song-artist pairs
            prompt_template = PromptTemplate.from_template(
            """
            You are a music recommendation assistant. The user is asking for songs similar to the following song-artist pairs:

            {query_for_langchain}
            
            Below are descriptions of songs that were retrieved from a song database based on these song-artist pairs:
            
            {combined_retrieved_data}
            
            Your task is to refine the query by focusing on the **common characteristics** shared between these songs. Analyze and describe the following aspects:

            - Genre
            - Tempo
            - Mood
            - Vocal style
            - Instrumentation
            - Era
            
            **Make sure that all recommended songs are from the 2000s, 2010s or later.** The recommendations should explore and include similar characteristics across different artists and genres, and avoid focusing too much on a single song or artist.

            Specifically, identify **common traits** such as energy level (high-energy or calm), instruments used (e.g., guitar, synthesizer, piano), vocal type (e.g., male or female vocals, solo or group), production style (e.g., acoustic, electronic), and mood (e.g., upbeat, melancholic, nostalgic).

            Then, refine the query to suggest songs with similar **overall characteristics**, even if they are from different artists or slightly different genres. Ensure that your refined query combines the most important attributes from each song in a balanced manner, without focusing exclusively on one artist or song. 

            **Important**: The refined query should list out the combined traits of the songs and provide a diverse set of recommendations. Output the final refined query in **one well-structured sentence**.

            FORMAT:
            - **Refined Query**: Find songs from the 2010s or later that are <refined characteristics>, featuring <key shared features>, and explore <themes/moods>.
            """
        )

            # Step 5: Format the prompt with the combined query and retrieved data
            prompt = prompt_template.format(query_for_langchain=query_for_langchain, combined_retrieved_data=combined_retrieved_data)

            logging.info(f"Prompt template: {prompt}")
            
            # Step 6: Use the LLM to refine the query
            response = await self.asyncOpenai.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": prompt}],
                response_format=RefineQuery,
            )

            parsed_result = response.choices[0].message.parsed

            # Step 7: Return the refined query
            return parsed_result.refined_query
        except Exception as e:
            logger.error(f"Error handling multiple song-artist query: {e}")
            return None

    async def get_relevant_documents(self, query: str, k: int = 10):
        try:
            return await self.vectorstore.as_retriever().aget_relevant_documents(query, k=k)
        except Exception as e:
            logger.error(f"Error fetching relevant documents: {e}")
            return []

    async def run(self, query: str):
        """
        Main method to handle various types of user inputs and return recommendations.
        """
        try:
            try:
                logging.info(f"Received query: {query}")
                query_type, Results = await self.determine_query_type(query)
                logging.info(f"Results: {Results}")
                if query_type == "single_song_artist":
                    return await self.handle_single_song_artist(Results.song_name, Results.artist_name)
            except Exception as e:
                logger.error(f"Failed to determine query type: {e}")
                return FunctionCallingWithTypesResponse(songInfoId=[])

            # logging.info(f"Refined query: {results}")
            # answers = await self.vectorstore.asimilarity_search(results, k=10, expr="MR == False")
            song_info_ids = []
            # for answer in answers:
            #     song_info_ids.append(int(answer.metadata['song_info_id']))
            return {
                "song_info_ids": song_info_ids,
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
                return FunctionCallingWithTypesResponse(songInfoId=[])
            return FunctionCallingWithTypesResponse(songInfoId = search_results["song_info_ids"])
                
        except Exception as e:
            logger.error(f"Error during GetFunctionCallingWithTypesRecommendation: {e}")
            logger.error(traceback.format_exc())  # 전체 스택 트레이스를 출력합니다.
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return FunctionCallingWithTypesResponse(songInfoId=[])