import os
import logging
import grpc
from langchain.agents import AgentExecutor, LLMSingleActionAgent, Tool, AgentOutputParser
from pymilvus import Collection, connections
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain_milvus import Milvus
from typing import List, Union
from langchain_openai import OpenAIEmbeddings
from service.AgentPromptTemplate import AgentPromptTemplate
from service.customAgentOutputParser import CustomAgentOutputParser
import traceback
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2 import LangchainAgentResponse, LangchainAgentRequest, SearchResult
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2_grpc import LangchainAgentRecommendServicer


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LangChainServiceAgentGrpc(LangchainAgentRecommendServicer):
    def __init__(self):
        # Load API keys from environment
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.milvus_host = os.getenv("MILVUS_HOST", "milvus-standalone")
        self.collection_name = "singsongsangsong_22286"
        connections.connect(alias="default", host=self.milvus_host, port="19530")
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
        
        # Define tools
        self.tools = [
            Tool(
                name="retriever",
                func=lambda query: self.retriever.get_relevant_documents(query),  # Wrap func in lambda
                description="""Use this to retrieve relevant documents from the Milvus collection. For example, 'songs similar to BTS songs', 'songs about break up' or 'dance songs which are relased around 2010s."""
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
            agent=self.agent, tools=self.tools, verbose=True, max_iterations=5
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

        # Convert and print the parsed response
        song_list, reason_list = self.parse_response_to_json(response)
 
        return song_list, reason_list

    def GetLangchainAgentRecommendation(self, request, context):
        try:
            # 유사한 노래 검색 (song_info_id 리스트가 반환됨)
            search_song_list, reason_list = self.run(request.command)  # 유사한 노래 검색 (song_info_id 리스트)
            logger.info(f"Search results: {search_song_list}")

            searchResult = []
            for i in range(len(search_song_list)):
                searchResult.append(SearchResult(songInfoId=int(search_song_list[i]), reason=reason_list[i]))

            return LangchainAgentResponse(searchResult=searchResult)
                
        except Exception as e:
            logger.error(f"Error during GetLangchainAgentRecommendation: {e}")
            logger.error(traceback.format_exc())  # 전체 스택 트레이스를 출력합니다.
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return LangchainAgentResponse(searchResult=[])