from langchain.vectorstores import Milvus
from langchain.embeddings.openai import OpenAIEmbeddings
from pymilvus import Collection
import os

class LangChainService:
    def __init__(self):
        # OpenAI Embeddings 설정 (여기서 OPENAI_API_KEY는 환경변수로 설정)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        
        # Milvus 설정
        self.collection_name = "singsongsangsong_22286"
        self.vectorstore = Milvus(
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
            connection_args={"host": "milvus-standalone", "port": "19530"}  # Milvus 서버에 맞게 설정
        )
        
    def search_similar_songs(self, query: str, top_k: int = 5):
        """입력된 query를 기반으로 가장 비슷한 노래를 추천"""
        # Query 임베딩 생성
        query_embedding = self.embeddings.embed_query(query)
        
        # Milvus에서 유사한 벡터 검색
        results = self.vectorstore.similarity_search(query_embedding, k=top_k)
        
        # song_info_id 반환 (유사한 결과의 ID 목록)
        song_ids = [result.metadata["song_info_id"] for result in results]
        return song_ids