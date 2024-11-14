import os
import logging
import grpc
import time
import aiomysql
import traceback
from operator import itemgetter
from pymilvus import Collection, connections
from langchain.agents import Tool
from langchain_openai.chat_models import ChatOpenAI
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from service.AgentPromptTemplate import AgentPromptTemplate
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2 import LangchainAgentResponse, SearchResult
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2_grpc import LangchainAgentRecommendServicer

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MySQLTool:
    def __init__(self, host, user, password, database, port=3306):
        self.db_host = host
        self.db_user = user
        self.db_password = password
        self.db_database = database
        self.db_port = port
        self.pool = None

    async def connect(self):
        """Initialize the connection pool."""
        try:
            self.pool = await aiomysql.create_pool(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                db=self.db_database,
                port=self.db_port,
                charset='utf8mb4',
                cursorclass=aiomysql.DictCursor,
                autocommit=True  # Enable autocommit for simplicity
            )
            logger.info("DB 연결 성공")
        except Exception as e:
            logger.error(f"DB 연결 실패: {e}")
            raise

    async def query(self, sql_query):
        """Execute a query and return the results."""
        if not self.pool:
            raise Exception("DB 연결이 초기화되지 않았습니다. connect()를 호출하세요.")

        async with self.pool.acquire() as connection:
            async with connection.cursor() as cursor:
                try:
                    await cursor.execute(sql_query)
                    result = await cursor.fetchall()
                    return result
                except Exception as e:
                    logger.error(f"쿼리 실패: {e}")
                    return {"error": str(e)}

    async def get_table_metadata(self, table_name):
        """Fetch column information for the given table."""
        if not self.pool:
            raise Exception("DB 연결이 초기화되지 않았습니다. connect()를 호출하세요.")

        sql_query = f"""
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY, COLUMN_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = '{self.db_database}' AND TABLE_NAME = '{table_name}';
        """
        async with self.pool.acquire() as connection:
            async with connection.cursor() as cursor:
                try:
                    await cursor.execute(sql_query)
                    columns = await cursor.fetchall()
                    return columns
                except Exception as e:
                    logger.error(f"메타데이터 조회 실패: {e}")
                    return {"error": str(e)}

    async def close(self):
        """Close the connection pool."""
        if self.pool:
            self.pool.close()
            await self.pool.wait_closed()
            logger.info("DB 연결 종료")

class LangChainServiceAgentGrpc(LangchainAgentRecommendServicer):
    def __init__(self):
        # Load API keys and initialize Milvus connection
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.milvus_host = os.getenv("MILVUS_HOST", "milvus-standalone")
        self.collection_name = "final_song_embeddings"

        connections.connect(alias="default", host=self.milvus_host, port="19530")
        self.collection = Collection(self.collection_name)

        # Initialize embeddings and vectorstore
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.vectorstore = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"host": self.milvus_host, "port": "19530"},
            text_field="song_name"
        )
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 10})  # Max 10 results

        # Initialize LLM
        self.llm = ChatOpenAI(
            temperature=0.5,
            max_tokens=4096,
            model_name="gpt-4o-mini",
            api_key=self.OPENAI_API_KEY,
        )

        # Initialize cache
        self.llm_cache = InMemoryCache()

        self.mysql_tool = None
        self.tools = []
        self.rag_chain = None
        self.agent_prompt = None
    
    async def initialize(self):
        try:
            set_llm_cache(self.llm_cache)
            # Initialize MySQL Tool
            self.mysql_tool = MySQLTool(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_DATABASE"),
            )
            await self.mysql_tool.connect()

            # Get table metadata
            table_name = "song_info"
            columns_metadata = await self.mysql_tool.get_table_metadata(table_name)
            columns_description = "\n".join(
                f"{col['COLUMN_NAME']} ({col['DATA_TYPE']}, Nullable: {col['IS_NULLABLE']}, Key: {col['COLUMN_KEY']})"
                for col in columns_metadata
            )

            # Define tools
            self.tools = [
                Tool(
                    name="retriever",
                    func=lambda query: self.retriever.get_relevant_documents(query),
                    description=(
                        "Retrieve relevant documents from the Milvus collection using vector cosine similarity. "
                        "Ideal for finding items based on semantic similarity, such as songs with similar characteristics or mood."
                    ),
                ),
                Tool(
                    name="mysql_query",
                    func=lambda query: self.mysql_tool.query(query),
                    description=f"Query the '{table_name}' table with the following columns: {columns_description}",
                ),
            ]

            # Define prompt template
            self.agent_prompt = AgentPromptTemplate(
                template=AgentPromptTemplate.agent_prompt_template,
                tools=self.tools,
                input_variables=["input", "intermediate_steps", "context", "question"],
            )

            # Define RAG chain
            def format_docs(args):
                docs = args["source_docs"]
                return "\n\n".join(doc.page_content for doc in docs)

            self.rag_chain = (
                RunnablePassthrough.assign(
                    input=lambda x: x["question"] if isinstance(x["question"], str) else str(x["question"]),
                    question=lambda x: x["question"] if isinstance(x["question"], str) else str(x["question"]),
                )
                | {
                    "source_docs": itemgetter("input") | self.retriever,
                    "question": itemgetter("question"),
                    "intermediate_steps": lambda _: [],  # 기본값 설정
                }
                | {
                    "source_docs": itemgetter("source_docs"),
                    "answer": {
                        "context": format_docs,
                        "question": itemgetter("question"),
                        "intermediate_steps": itemgetter("intermediate_steps"),
                        "input": itemgetter("question"),
                    }
                    | self.agent_prompt
                    | self.llm
                    | StrOutputParser()
                }
            )
            logger.info("LangChainServiceAgentGrpc initialized successfully")
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def ainvoke(self, question):
        try:
            if not isinstance(question, str):
                raise ValueError(f"Question must be a string, but got {type(question)}")
            
            logger.info(f"Invoking RAG Chain with question: {question}")
            response = await self.rag_chain.ainvoke({"question": question})

            if not response:
                logger.error(f"No response for question: {question}")
                return None

            return response
        except Exception as e:
            logger.error(f"Error invoking RAG Chain: {e}")
            return None

    def parse_response_to_json(self, response):
        try:
            # 응답이 문자열인지 확인
            if isinstance(response, str):
                song_info_ids = []
                reasons = []

                # 문자열을 줄 단위로 분리하여 파싱
                for s in response.split("\n"):
                    if "song_info_id" in s:
                        song_info_ids.append(s.split(":")[-1].strip())
                    elif "reason" in s:
                        reasons.append(s.split(":")[-1].strip())
                return song_info_ids, reasons

            # 응답이 dict 구조일 경우
            if isinstance(response, dict):
                # `source_docs`에서 song_info_id와 reason 추출
                source_docs = response.get("source_docs", [])
                answers = response.get("answer", "")
                song_info_ids = [doc.metadata.get("song_info_id") for doc in source_docs]
                reasons = []

                # 응답의 answer 부분에서 reason 추출
                for line in answers.split("\n"):
                    if "reason" in line:
                        reasons.append(line.split(":")[-1].strip())

                return song_info_ids, reasons

            logger.error(f"Unsupported response format: {type(response)}")
            return [], []

        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return [], []

    async def run(self, query):
        """
        Executes a query using the executor and returns the response.
        Args:
            query (str): The query to be executed.
        Returns:
            Tuple[List[str], List[str]]: A list of song_info_ids and reasons.
        """
        try:
            # Run the agent
            response = await self.ainvoke(query)
            if not response:
                logger.error(f"No response for query: {query}")
                return [], []
        
            print(response)

            # Parse and return the response
            song_info_ids, reasons = self.parse_response_to_json(response)
            return song_info_ids, reasons
        except Exception as e:
            logger.error(f"Failed to run query: {e}")
            return [], []

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