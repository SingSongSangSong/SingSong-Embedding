import os
import logging
import grpc
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool
from pymilvus import Collection, connections
from langchain.chains import LLMChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from service.AgentPromptTemplate import AgentPromptTemplate
from service.customAgentOutputParser import CustomAgentOutputParser
import traceback
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2 import LangchainAgentResponse, SearchResult
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2_grpc import LangchainAgentRecommendServicer
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
import time
from dotenv import load_dotenv
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainServiceAgentGrpc(LangchainAgentRecommendServicer):
    def __init__(self, config, collection, embedding_model, retriever, vectorstore, llm):
        # Load API keys from environment
        self.OPENAI_API_KEY = config["OPENAI_API_KEY"]
        self.milvus_host = config["MILVUS_HOST"]
        self.collection_name = config["COLLECTION_NAME"]
        self.collection = collection
        self.embedding_model = embedding_model
        self.retriever = retriever
        self.vectorstore = vectorstore
        self.retriever = retriever
        self.llm = llm
        self.llm_cache = InMemoryCache()
        set_llm_cache(self.llm_cache)
        
        # Define tools
        self.tools = [
            Tool(
                name="retriever",
                func=lambda query: self.retriever.get_relevant_documents(query),  # Wrap func in lambda
                description="""Use this to retrieve relevant documents from the Milvus collection using vector cosine similarity. all the vector contains song information such as song name, artist name, genre, year, country, artist type, artist gender, lyrics summary, tags and stuff.""",
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
        

    async def run(self, query):
        """
        Executes a query using the executor and returns the response.
        Args:
            query (str): The query to be executed.
        Returns:
            str: The response generated by the agent.
        """
        # Run the agent
        response = await self.executor.arun(query)
        print("Response: ", response)

        # Convert and print the parsed response
        song_list, reason_list = self.parse_response_to_json(response)
 
        return song_list, reason_list

    async def GetLangchainAgentRecommendation(self, request, context):
        try:
            # 유사한 노래 검색 (song_info_id 리스트가 반환됨)
            start_time = time.time()
            search_song_list, reason_list = await self.run(request.command)  # 유사한 노래 검색 (song_info_id 리스트)
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