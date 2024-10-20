import os
import logging
import grpc
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from pymilvus import Collection, connections
from langchain_openai.chat_models import ChatOpenAI
from langchain_milvus import Milvus
from langchain.chains import LLMChain
from langchain_openai import OpenAIEmbeddings
from service.AgentPromptTemplateForTypes import AgentPromptTemplate
from service.customAgentOutputParser import CustomAgentOutputParser
import traceback
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2 import LangchainAgentResponse, SearchResult
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2_grpc import LangchainAgentRecommendServicer
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
import time
from openai import OpenAI, AsyncOpenAI
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from typing import List, ClassVar

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAiFunctionOutput(BaseModel):
    song_info_id: List[str]
    reason: List[str]

class ExtractCommonTraits(BaseModel):
    genre: str
    year: str
    country: str
    artist_type: str
    artist_gender: str
    situation: List[str]
    octave: str
    vocal_range: str
    artist_name: List[str]
    lyrics: str

class ExtractInfoFromQuery(BaseModel):
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

class LangChainServiceAgentGrpc(LangchainAgentRecommendServicer):
    def __init__(self):
        # Load API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.milvus_host = os.getenv("MILVUS_HOST", "milvus-standalone")
        self.collection_name = "singsongsangsong_22286"
        self.collection = Collection(self.collection_name)
        # Define embedding model
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        # Milvus vector store
        self.vectorstore = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"host": self.milvus_host, "port": "19530"},
            text_field="song_name"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})  # Max 10 results
        # Initialize LLM model
        self.llm = ChatOpenAI(
            temperature=0.5,
            max_tokens=4096,
            model_name='gpt-4o-mini',
            api_key=self.OPENAI_API_KEY
        )
        self.asyncOpenAi = AsyncOpenAI(api_key=self.OPENAI_API_KEY)
        self.openai = OpenAI(api_key=self.OPENAI_API_KEY)
        
        # Define tools
        self.tools = [
            Tool(
                name = "Search_Ten_Items_In_MilvusDB_with_Similiarity",
                func=lambda query: self.search_in_milvusdb(query),
                description="Useful when you want to search for similar items in MilvusDB and return the top 10 results. For Example: 'song similar to BTS' or 'song similar to BTS Dynamite' or 'song name similar to Butter'"
            ),
            Tool(
                name = "Extract_Information_From_Query",
                func=lambda query: self.extract_information_from_query(query),
                description="Useful when you want to extract information from a query. Such as Song Name, Artist Name, Octave, Vocal Range, year, country, genre, situation, etc."
            ),
            Tool(
                name = "Refine_query",
                func=lambda data: self.ExtractCommonTraitService(data),
                description="Useful when you want to extract common feataures for several songs such as genre, year, country, artist type, artist gender, situation, octave, vocal range, artist name, lyrics."
            )
        ]
        # Define prompt template
        self.agent_prompt = AgentPromptTemplate(
            template=AgentPromptTemplate.agent_prompt_template,
            tools=self.tools,
            input_variables=["input", "intermediate_steps"],
        )
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.output_parser = CustomAgentOutputParser()
        # Create LLM chain
        llm_chain = LLMChain(llm=self.llm, prompt=self.agent_prompt)
        tool_names = [tool.name for tool in self.tools]
        # Create single action agent
        self.agent = LLMSingleActionAgent(
            llm_chain=llm_chain, output_parser=self.output_parser, stop=["\nObservation:"], allowed_tools=tool_names
        )
        # Create executor
        self.executor = AgentExecutor.from_agent_and_tools(
            agent=self.agent, tools=self.tools, verbose=True, max_iterations=2
        )

    def parse_response_to_json(self, response: str):
        song_info_ids = []
        reasons = []

        for s in response.split("\n"):
            if "song_info_id" in s:
                song_info_ids.append(s.split(":")[-1].strip())
            elif "reason" in s:
                reasons.append(s.split(":")[-1].strip())
        return song_info_ids, reasons

    def search_in_milvusdb(self, query):
        return self.retriever.invoke(query)
    
    def extract_information_from_query(self, query):
        messages = [
            {"role": "system", "content": """
            You are an assistant that categorizes user inputs into specific types of song recommendations and extracts important information.

            Follow these guidelines for extracting relevant information, ensuring all relevant fields are extracted even if the information is not provided.
            Always extract all the relevant fields, even if the information is missing, by returning `None` or an empty list where applicable.

            Format:
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
        response = self.openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=messages,
            response_format=ExtractInfoFromQuery,
        )

        # Extract and return query type
        parsed_result = response.choices[0].message.parsed
        return parsed_result

    def ExtractCommonTraitService(self, data):
        # Step 4: Create the detailed prompt for multiple song-artist pairs
        prompt_template = PromptTemplate.from_template(
            """
            you are a music recommendation assistant. The user is asking for several songs so you have to extract the common features and create the query based on that.

            {data}

            You have to extract the common features and gotta suggest what kinds of feature that users want to get. You have to consider the following aspects:

            - Genre(국악, 발라드, 록/메탈, 댄스, 성인가요/트로트, 포크/블루스, 키즈, 창작동요, 국내영화, 국내드라마, 랩/힙합, R&B/Soul, 인디음악, 애니메이션/웹툰, 만화, 교과서동요, 국외영화, POP, 클래식, 크로스오버, J-POP, CCM, 게임, 컨트리, 재즈, 보컬재즈, 포크, 블루스, 일렉트로니카, 월드뮤직, 애시드/퓨전/팝, 국내뮤지컬)
            - Year
            - Country
            - Artist Type
            - Artist Gender
            - Situation(Classics, Ssum, Breakup, Carol, Finale, Dance, Duet, Rainy, Office, Wedding, Military)
            - Octave
            - Vocal Range
            - Artist Name
            - Lyrics

            Format:
            - Genre: <genre>
            - Year: <year>
            - Country: <country>
            - Artist Type: <artist_type>
            - Artist Gender: <artist_gender>
            - Situation: [<situation>] (Classics, Ssum, Breakup, Carol, Finale, Dance, Duet, Rainy, Office, Wedding, Military)
            - Octave: <octave> 
            - Vocal Range: <vocal_range>
            - Artist Name: <artist_name>
            - Lyrics: <lyrics>
            """
        )
        try:
            # Step 5: Format the prompt with the combined query and retrieved data
            prompt = prompt_template.format(query_for_langchain=data)
            logging.info(f"Prompt template: {prompt}")

            # Step 6: Use the LLM to refine the query
            response = self.openai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format=ExtractCommonTraits,
            )
            parsed_result = response.choices[0].message.parsed
            # Step 7: Return the refined query
            return parsed_result
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            return None

    def run(self, query):
        """
        Executes a query using the executor and returns the response.
        Args:
            query (str): The query to be executed.
        Returns:
            str: The response generated by the agent.
        """
        # Run the agent
        response = self.executor.run(query)
        print("Response: ", response)

        # Convert and print the parsed response
        song_list, reason_list = self.parse_response_to_json(response)
 
        return song_list, reason_list

    def GetLangchainAgentRecommendation(self, request, context):
        try:
            # 유사한 노래 검색 (song_info_id 리스트가 반환됨)
            start_time = time.time()
            search_song_list, reason_list = self.run(request.command)  # 유사한 노래 검색 (song_info_id 리스트)
            logger.info(f"Search results: {search_song_list}")

            searchResult = []
            for i in range(len(search_song_list)):
                searchResult.append(SearchResult(songInfoId=int(search_song_list[i]), reason=reason_list[i]))
            
            logger.info(f"GetLangchainAgentRecommendation took {time.time() - start_time} seconds")

            return LangchainAgentResponse(searchResult=searchResult)
                
        except Exception as e:
            logger.error(f"Error during GetLangchainAgentRecommendation: {e}")
            logger.error(traceback.format_exc())  # 전체 스택 트레이스를 출력합니다.
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return LangchainAgentResponse(searchResult=[])