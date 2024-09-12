from langchain_milvus import Milvus
from langchain_openai.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.chains.transform import TransformChain
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from proto.langchainRecommend.langchainRecommend_pb2_grpc import LangchainRecommendServicer
from proto.langchainRecommend.langchainRecommend_pb2 import LangchainResponse, SimilarItem
import grpc
import traceback
import logging
import os

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainServiceGrpc(LangchainRecommendServicer):
    def __init__(self):
        # Load API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        # Initialize LLM model
        self.llm = ChatOpenAI(
            temperature=0.5,
            max_tokens=4096,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_name='gpt-4o-mini',
            api_key=self.OPENAI_API_KEY
        )

        connections.connect(alias="default", host="milvus-standalone", port="19530")
        self.collection = Collection("singsongsangsong_22286")

        # Embedding model for user profiles
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

        # Milvus 설정
        self.collection_name = "singsongsangsong_22286"
        self.vectorstore = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"host": "milvus-standalone", "port": "19530"},
            text_field="song_name"
        )
        self.retriever = self.vectorstore.as_retriever()

    # 1. 노래 제목과 가수 이름 기반 검색 (유형 1)
    def search_by_song_and_artist(self, output):
        extracted_info = output["extracted_info"]

        artist_song_pairs = extracted_info.split(", ")
        search_results = []

        for pair in artist_song_pairs:
            artist_name = pair.split("Artist:")[-1].split(",")[0].strip()
            song_title = pair.split("Song:")[-1].strip()
            search_query = f"{artist_name} released the song '{song_title}'"
            results = self.retriever.get_relevant_documents(search_query)

            search_results += [doc.metadata['song_info_id'] for doc in results]

        search_results = self.ensure_five_recommendations(search_results)
        return {"retrieved_data": search_results}

    # 2. 기분이나 분위기 기반 검색 (유형 2)
    def search_by_mood_or_theme(self, user_query):
        prompt = f"""
        The user input is: "{user_query}". 
        Extract the mood or theme from the input and create a query to search for songs with similar emotional qualities, such as mood, energy level, and use case.
        """
        refined_query = self.llm(prompt)
        results = self.retriever.get_relevant_documents(refined_query.content)
        
        search_results = [doc.metadata['song_info_id'] for doc in results]
        search_results = self.ensure_five_recommendations(search_results)
        return {"retrieved_data": search_results}

    # 3. 특정 연도나 특성 기반 검색 (유형 3)
    def search_by_specific_feature(self, user_query):
        prompt = f"""
        The user is looking for songs with the following features: "{user_query}". 
        Create a detailed query that includes era, singer type, and genre to retrieve relevant songs from the database.
        """
        refined_query = self.llm(prompt)
        results = self.retriever.get_relevant_documents(refined_query.content)
        
        search_results = [doc.metadata['song_info_id'] for doc in results]
        search_results = self.ensure_five_recommendations(search_results)
        return {"retrieved_data": search_results}

    # 추천 목록을 정확히 5곡으로 맞추는 함수
    def ensure_five_recommendations(self, recommendation_list):
        while len(recommendation_list) < 5:
            recommendation_list.append(recommendation_list[-1])  # 마지막 곡을 반복 추가
        return recommendation_list[:5]

    # LLM을 이용해 입력 타입을 판단하는 함수
    def determine_input_type(self, user_query):
        prompt = f"""
        Classify the user's input into one of the following types:
        1. song_and_artist: If the user is asking for songs similar to specific songs or artists.
        2. mood_or_theme: If the user is describing a mood, theme, or situation to find songs that match those qualities.
        3. specific_feature: If the user is looking for songs based on specific features such as year, genre, or singer type.

        Input: "{user_query}"

        Your output should be in the following format:
        Type: <song_and_artist/mood_or_theme/specific_feature>
        """
        result = self.llm(prompt)
        input_type = result.content.lower().replace("type:", "").strip()
        return input_type

    # LLMChain을 통해 검색 및 추천 수행
    def run_chain(self, user_query):
        input_type = self.determine_input_type(user_query)

        if input_type == "song_and_artist":
            transform_chain = TransformChain(
                input_variables=["extracted_info"],
                output_variables=["retrieved_data"],
                transform=self.search_by_song_and_artist
            )
        elif input_type == "mood_or_theme":
            transform_chain = TransformChain(
                input_variables=["query"],
                output_variables=["retrieved_data"],
                transform=self.search_by_mood_or_theme
            )
        elif input_type == "specific_feature":
            transform_chain = TransformChain(
                input_variables=["query"],
                output_variables=["retrieved_data"],
                transform=self.search_by_specific_feature
            )
        else:
            raise ValueError(f"Unrecognized input type: {input_type}")

        prompt_extract = ChatPromptTemplate.from_template(
            """
            Extract the artist name and song title from the following query: {query}.
            If no song or artist is found, try to extract the mood, theme, or other musical characteristics from the query.

            Your output should follow this format:
            - Artist: <artist_name>
            - Song: <song_title> OR Mood/Theme: <mood_or_theme>
            """
        )

        prompt_refine = ChatPromptTemplate.from_template(
            """
            Based on the query "{query}" and the following song metadata retrieved from the database:
            {retrieved_data}

            Please provide exactly 5 song recommendations by their song info IDs, formatted as:

            - Recommendation: [<song_info_id1>, <song_info_id2>, <song_info_id3>, <song_info_id4>, <song_info_id5>]

            Only provide the song info IDs without any additional descriptions or explanations.
            """
        )

        llm_chain1 = LLMChain(llm=self.llm, prompt=prompt_extract, output_key="extracted_info")
        llm_chain2 = LLMChain(llm=self.llm, prompt=prompt_refine, output_key="final_response")

        sequential_chain = SequentialChain(
            chains=[llm_chain1, transform_chain, llm_chain2],
            input_variables=["query"],
            output_variables=["final_response"]
        )

        response = sequential_chain.invoke({"query": user_query})
        return response['final_response']

    def parse_recommendation(self, recommendation_str):
        try:
            # "Recommendation: [14032, 46439, 14059, 14093, 14093]"에서 리스트 부분만 추출
            recommendation_str = recommendation_str.split("Recommendation: ")[1].strip()

            # 리스트 문자열에서 대괄호 제거하고, 쉼표로 구분된 값을 int로 변환
            recommendation_list = [int(x.strip()) for x in recommendation_str.strip('[]').split(',')]

            return recommendation_list
        except IndexError:
            # "Recommendation: " 부분이 없을 때 예외 처리
            raise ValueError("Invalid format: 'Recommendation: ' keyword not found")
        except ValueError as e:
            # 숫자가 아닌 값이 리스트에 있을 경우 예외 처리
            raise ValueError(f"Invalid number format in recommendation: {str(e)}")

    def GetLangchainRecommendation(self, request, context):
        try:
            # 유사한 노래 검색 (song_info_id 리스트가 반환됨)
            search_results = self.run_chain(request.command)  # 유사한 노래 검색 (song_info_id 리스트)
            logger.info(f"Search results: {search_results}")

            # recommendation 문자열을 파싱하여 int64 리스트로 변환
            int64_list = self.parse_recommendation(search_results)

            similar_items = []

            # Milvus에서 song_info_id에 해당하는 데이터 조회
            search_results = self.collection.query(
                expr=f"song_info_id in {int64_list}",  # song_info_id 리스트에 있는 값으로 필터링
                output_fields=["description", "song_info_id", "artist_name", "MR", "ssss", "audio_file_url", "album", "song_number", "song_name"]
            )
            logger.info(f"Retrieved data: {search_results}")

            # 검색 결과 순회
            for result in search_results:
                logger.info(f"Song Info ID: {result.get('song_info_id')}")

                # SimilarItem 메시지 생성 및 추가
                similar_items.append(SimilarItem(
                    songInfoId=result.get("song_info_id"),  # song_info_id
                    songName=result.get("song_name"),  # 노래 제목
                    singerName=result.get("artist_name"),  # 아티스트 이름
                    isMr=result.get("MR", False),  # MR 여부
                    ssss=result.get("ssss", ""),  # 추가 메타데이터 필드
                    audioFileUrl=result.get("audio_file_url", ""),  # 오디오 파일 URL
                    album=result.get("album", ""),  # 앨범 이름
                    songNumber=result.get("song_number", 0),  # 곡 번호
                    similarityScore=1.0  # 유사도 점수는 하드코딩했지만, 필요시 계산 가능
                ))

            return LangchainResponse(similarItems=similar_items)
                
        except Exception as e:
            logger.error(f"Error during GetLangchainRecommendation: {e}")
            logger.error(traceback.format_exc())  # 전체 스택 트레이스를 출력합니다.
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return LangchainResponse(similarItems=[])