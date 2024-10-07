import os
import logging
from pydantic import BaseModel
from openai import OpenAI
from pymilvus import Collection, connections
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai.chat_models import ChatOpenAI
from typing import List
from proto.functionCallingRecommend.functionCallingRecommend_pb2 import FunctionCallingResponse
from proto.functionCallingRecommend.functionCallingRecommend_pb2_grpc import functionCallingRecommendServicer
import grpc
import traceback
import logging
import os

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

class RefineQuery(BaseModel):
    refined_query: str

class FunctionCallingServiceGrpc(functionCallingRecommendServicer):
    def __init__(self):
        # Load API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.collection_name = "singsongsangsong_22286"
        connections.connect(alias="default", host=self.milvus_host, port="19530")
        self.collection = Collection(self.collection_name)

        # Define embedding model
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.openai = OpenAI(api_key=self.OPENAI_API_KEY)

        # Initialize LLM model
        self.llm = ChatOpenAI(
            temperature=0.5,
            max_tokens=4096,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_name='gpt-4o-mini',
            api_key=self.OPENAI_API_KEY
        )

        # Milvus vector store
        self.vectorstore = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"host": self.milvus_host, "port": "19530"},
            text_field="song_name"
        )
        self.retriever = self.vectorstore.as_retriever()  # Max 10 results

    def determine_query_type(self, query: str):
        """
        Ask OpenAI to determine the type of query (specific song/artist, mood/genre, or specific feature).
        """
        messages = [
            {"role": "system", "content": """
            You are an assistant that categorizes user inputs into specific types of song recommendations.

            Follow these guidelines for classifying the query into one of three categories:

            1. **Single Song/Artist Query:**
                - If the user input contains a specific song title and artist name, this is a query that asks for songs similar to the provided song or artist. 
                - For example, "Recommend me songs similar to 'Bang Bang Bang' by BigBang" or "Songs similar to Buzz's 'Thorn'".
                - Output query type as 'single_song_artist'.

            2. **Multiple Song-Artist Pairs Query:**
                - If the user provides multiple song-artist pairs, your job is to analyze the common features (e.g., shared genres, moods, or musical styles) between the songs and artists. 
                - For example, if the user says, "Recommend me songs like 'Bang Bang Bang' by BigBang and 'Thorn' by Buzz", this query is asking for common characteristics between the two or more songs.
                - Output query type as 'multiple_song_artist'.

            3. **Mood/Feature-based Query:**
                - If the user does not mention specific songs or artists but instead provides a mood, feeling, or general theme, this is a mood or feature-based query. 
                - For example, "I'm feeling down, recommend me some sad songs" or "Recommend me energetic songs for a workout."
                - Output query type as 'mood_feature'.

            It's important to ensure that the query fits into only one of these three categories. If the user input is unclear, make your best effort to infer the most likely category. 
            Once you have decided, return the type of query as either 'single_song_artist', 'multiple_song_artist', or 'mood_feature'.
            If the type is 'single_song_artist' or 'multiple_song_artist', you should also extract the song names and artist names from the query.
             
            Format
            - Query Type: <single_song_artist/multiple_song_artist/mood_feature>
            - Song Name: [<song_name1>, <song_name2>, ...]
            - Artist Name: [<artist_name1>, <artist_name2>, ...]
            """},
            {"role": "user", "content": query}
        ]

        # Use OpenAI to determine query type
        response = self.openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=QueryType,
        )

        # Extract and return query type
        parsed_result = response.choices[0].message.parsed
        query_type = parsed_result.query_type
        song_names = parsed_result.song_name
        artist_names = parsed_result.artist_name
        return query_type, song_names, artist_names

    def handle_single_song_artist(self, song_name: str, artist_name: str):
        """
        Handles queries to find similar songs based on a song and artist using Milvus DB and LCEL-style chain.
        """
        # Create the query
        query = f"Find songs similar to {song_name} by {artist_name}"

        # Retrieve relevant documents from Milvus (assuming `retriever` is set up for Milvus)
        results = self.vectorstore.similarity_search(query, k=1, expr="MR == False")  # Example retriever setup
        
        # Combine the retrieved document descriptions into one text block
        retrieved_data = "\n".join([doc.metadata.get("description") for doc in results])

        # Create the prompt using the retrieved data
        prompt_template = PromptTemplate.from_template(
            """
            You are a music recommendation assistant. The user is asking for songs similar to the following query: "{query}". Below are descriptions of songs that were retrieved from a song database based on this query:
            
            {retrieved_data}
            
            Your task is to refine the user's query by analyzing the common characteristics of these songs, such as their genre, tempo, mood, vocal style, instrumentation, and era. Ensure that the recommendations are not limited to songs by the same artist. 

            Specifically, you should consider shared attributes such as energy level (high-energy or calm), instruments used (e.g., guitar, synthesizer, piano), vocal type (e.g., male or female vocals, solo or group), production style (e.g., acoustic, electronic), and mood (e.g., upbeat, melancholic, nostalgic).

            Then, provide a refined query suggesting songs with similar overall characteristics, even if they are from different artists or slightly different genres.

            **Important**: You must output the refined query in **one single, well-structured sentence**. The format of the output must exactly follow the example below:

            - **Refined Query**: Find songs that are <refined characteristics>, featuring <key shared features>, and explore <themes/moods>.

            Here is an example of the expected format:
            
            - Refined Query: Find songs that are in the ballad genre, featuring male solo vocals with a calm and nostalgic mood, moderate tempo, and soft instrumental textures, exploring themes of love and breakups.
            """
        )

        # Format the template with actual retrieved data
        prompt = prompt_template.format(query=query, retrieved_data=retrieved_data)
        
        # Now, pass the formatted prompt (as a string) to the LLM
        response = self.openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format=RefineQuery,
        )

        parsed_result = response.choices[0].message.parsed
                
        # Parse the response and return the recommendations
        return parsed_result.refined_query

    def handle_multiple_song_artist(self, songs: list):
        """
        Handles queries with multiple song-artist pairs, finds common features and retrieves songs based on those.
        """
        # Step 1: Build the combined query for all the song-artist pairs
        query_for_langchain = "Find songs similar to the following songs:"
        for song in songs:
            query_for_langchain += f" '{song['song_name']}' by {song['artist_name']}."

        # Step 2: Retrieve relevant documents for each song
        documents = []
        for song in songs:
            query = f"Find songs similar to {song['song_name']} by {song['artist_name']}"
            results = self.get_relevant_documents(query, 1)
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
            
            Ensure that the recommendations are not limited to songs by the same artists. Explore and include similar characteristics across different artists and genres, and avoid focusing too much on a single song or artist.

            Specifically, identify **common traits** such as energy level (high-energy or calm), instruments used (e.g., guitar, synthesizer, piano), vocal type (e.g., male or female vocals, solo or group), production style (e.g., acoustic, electronic), and mood (e.g., upbeat, melancholic, nostalgic).

            Then, refine the query to suggest songs with similar **overall characteristics**, even if they are from different artists or slightly different genres. Ensure that your refined query combines the most important attributes from each song in a balanced manner, without focusing exclusively on one artist or song.

            **Important**: The refined query should list out the combined traits of the songs and provide a diverse set of recommendations. Output the final refined query in **one well-structured sentence**.

            FORMAT:
            - **Refined Query**: <refined_query>
            """
        )

        # Step 5: Format the prompt with the combined query and retrieved data
        prompt = prompt_template.format(query_for_langchain=query_for_langchain, combined_retrieved_data=combined_retrieved_data)
        
        # Step 6: Use the LLM to refine the query
        response = self.openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format=RefineQuery,
        )

        parsed_result = response.choices[0].message.parsed

        # Step 7: Return the refined query
        return parsed_result.refined_query

    def handle_mood_or_feature_query(self, query: str):
        """
        Handles queries based on mood, feature, theme, or specific situations.
        Refines the user's query by adding detailed musical elements like genre, mood, time period, or specific characteristics.
        """
        # Use OpenAI to refine the user's query into a more detailed, music-specific search query
        refined_query_message = [
            {"role": "system", "content": (
                "You are a music recommendation assistant. Your task is to take user input and refine it by adding detailed "
                "musical characteristics based on the context. Ensure the refined query includes genre, mood, tempo, or any relevant "
                "attributes that fit the user's request.\n"
                "Consider the following contexts:\n"
                "1. When the user requests recommendations based on a specific genre or mood, suggest additional details such as popular subgenres, "
                "instruments used, and energy level.\n"
                "2. When the user is looking for music to match emotions or situations (e.g., happy, nostalgic, rainy day), include suitable tempo, "
                "vocal style, and mood-appropriate genres.\n"
                "3. For thematic or weather-based requests, mention relevant genres, rhythms, and the atmosphere the music creates.\n"
                "4. If the user is asking for music from a specific time period (e.g., 'songs from the 90s'), consider the dominant genres, popular artists, and production styles from that era.\n"
                "5. For requests based on unique features (e.g., instrumental, high-energy, acoustic), include defining characteristics of those features in your refinement."

                "The output must be in a format like below:\n"
                "refiend query: <refined_query>"
            )},
            {"role": "user", "content": (
                f"User query: {query}\n"
                "Refine the above query by adding detailed musical elements like genre, mood, tempo, era, and any other relevant characteristics."
            )}
        ]

        response = self.openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=refined_query_message,
            response_format=RefineQuery,
        )
        refined_query = response.choices[0].message.parsed.refined_query
        # Use the refined query for Milvus search
        return refined_query

    def get_relevant_documents(self, query: str, k: int = 10):
        """
        Retrieves relevant documents from Milvus based on the query.
        """
        results = self.vectorstore.as_retriever().get_relevant_documents(query, k=k)
        return results

    def run(self, query: str):
        """
        Main method to handle various types of user inputs and return recommendations.
        """
        try:
            # Determine the type of query
            query_type, song_names, artist_names = self.determine_query_type(query)

            # Handle the query based on its type
            if query_type == "single_song_artist":
                # Example: 빅뱅의 뱅뱅뱅
                results = self.handle_single_song_artist(song_names[0], artist_names[0])

            elif query_type == "multiple_song_artist":
                # Example: 여러 노래와 가수의 공통점을 기반으로 추천
                songs = [
                    {"song_name": song, "artist_name": artist} for song, artist in zip(song_names, artist_names)
                ]
                results = self.handle_multiple_song_artist(songs)

            elif query_type == "mood_feature":
                # Example: 분위기나 테마에 따른 추천
                results = self.handle_mood_or_feature_query(query)
            else:
                logging.error("Unknown query type.")
                return None
            
            answers = self.vectorstore.similarity_search(results, k=10, expr="MR == False")
            song_info_ids = []
            for answer in answers:
                song_info_ids.append(int(answer.metadata['song_info_id']))
            return {
                "song_info_ids": song_info_ids,
            }

        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            return None
        
    def GetFunctionCallingRecommendation(self, request, context):
        try:
            # 유사한 노래 검색 (song_info_id 리스트가 반환됨)
            search_results = self.run(request.command)  # 유사한 노래 검색 (song_info_id 리스트)
            return FunctionCallingResponse(songInfoId = search_results["song_info_ids"])
                
        except Exception as e:
            logger.error(f"Error during GetFunctionCallingRecommendation: {e}")
            logger.error(traceback.format_exc())  # 전체 스택 트레이스를 출력합니다.
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return FunctionCallingResponse(songInfoId=[])