import grpc
import logging
import asyncio
import signal
import aiomysql
# from ddtrace import tracer
# from apscheduler.schedulers.asyncio import AsyncIOScheduler  # 비동기 스케줄러
from service.funtionCallingWithTypes import FunctionCallingWithTypesServiceGrpc
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2_grpc import add_FunctionCallingWithTypesRecommendServicer_to_server
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from openai import OpenAI, AsyncOpenAI
from pymilvus import Collection, connections
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from service.langchainServiceAgentGrpc import LangChainServiceAgentGrpc


load_dotenv()  # .env 로딩

# 환경변수 수동 주입
config = {
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
    "MILVUS_HOST": os.getenv("MILVUS_HOST"),
    "DB_HOST": os.getenv("DB_HOST"),
    "DB_USER": os.getenv("DB_USER"),
    "DB_PASSWORD": os.getenv("DB_PASSWORD"),
    "DB_DATABASE": os.getenv("DB_DATABASE"),
    "DB_PORT": int(os.getenv("DB_PORT", 30007)),
    "COLLECTION_NAME": "final_song_embeddings"
}

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


grpc_server = None

async def setup_db_config():
    try:
        # 비동기 MySQL 연결 설정
        pool = await aiomysql.create_pool(
            host=config["DB_HOST"],
            user=config["DB_USER"],
            password=config["DB_PASSWORD"],
            db=config["DB_DATABASE"],
            port=config["DB_PORT"],
            charset='utf8mb4',
            cursorclass=aiomysql.DictCursor,
            autocommit=False  # 자동 커밋 설정
        )
        logger.info("DB 연결 성공")
        return pool

    except aiomysql.MySQLError as e:
        logger.error(f"MySQL 연결 실패: {e}")
        raise


# gRPC 서버 실행
async def serve_grpc():
    global grpc_server
    grpc_server = grpc.aio.server()  # 비동기 gRPC 서버 생성

    # Milvus 연결
    connections.connect(alias="default", host=config["MILVUS_HOST"], port="19530")
    milvus_collection = Collection(config["COLLECTION_NAME"])  # 연결 후 컬렉션 인스턴스 생성
    # Embedding 모델 & OpenAI 객체
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    openai_client = OpenAI(api_key=config["OPENAI_API_KEY"])
    async_openai_client = AsyncOpenAI(api_key=config["OPENAI_API_KEY"])
    llm = ChatOpenAI(
            temperature=0.5,
            max_tokens=4096,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()],
            model_name='gpt-4o-mini',
            api_key=config["OPENAI_API_KEY"]
        )
    vectorstore = Milvus(
            embedding_function=embedding_model,
            collection_name=config["COLLECTION_NAME"],
            connection_args={"host": config["MILVUS_HOST"], "port": "19530"},
            text_field="song_name"
        )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})  # Max 10 results

    pool = await setup_db_config()  # DB 설정 비동기 초기화

    # 다른 서비스도 동일하게 처리
    langchainAgent = LangChainServiceAgentGrpc(config, milvus_collection, embedding_model, retriever, vectorstore, llm)
    service = FunctionCallingWithTypesServiceGrpc(config, milvus_collection, embedding_model, openai_client, async_openai_client, retriever, llm, langchainAgent, pool, vectorstore=vectorstore)

    add_FunctionCallingWithTypesRecommendServicer_to_server(service, grpc_server)

    grpc_server.add_insecure_port('[::]:50051')
    await grpc_server.start()
    logger.info("gRPC server started on port 50051")

# Graceful Shutdown 핸들러
async def shutdown():
    global grpc_server

    logger.info("Shutting down gracefully...")

    # gRPC 서버 중지
    if grpc_server:
        await grpc_server.stop(5)  # 최대 5초 동안 종료 대기
        logger.info("gRPC server stopped.")

    logger.info("Shutdown completed.")

# 신호 처리기 등록
def register_signal_handlers(loop):
    for signal_name in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signal_name),
            lambda: asyncio.ensure_future(shutdown())
        )

# 비동기 메인 함수
async def main():
    # tracer.configure(port=8126, https=False)

    # gRPC 서버 시작
    await serve_grpc()

    # gRPC 서버 종료 대기
    await grpc_server.wait_for_termination()

if __name__ == "__main__":
    # 이벤트 루프 가져오기
    loop = asyncio.get_event_loop()

    # 신호 처리기 등록
    register_signal_handlers(loop)

    # 비동기 메인 함수 실행
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("Event loop closed.")