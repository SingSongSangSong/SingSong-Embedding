import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import grpc
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2_grpc import LangchainAgentRecommendStub
from proto.langchainAgentRecommend.langchainAgentRecommend_pb2 import LangchainAgentRequest

def run():
    # gRPC 서버에 연결
    with grpc.insecure_channel('localhost:50051') as channel:
        # LangchainRecommendStub을 사용해 서버와 통신
        stub = LangchainAgentRecommendStub(channel)

        # LangchainRequest 생성
        request = LangchainAgentRequest(
            memberId=8,  # 예시로 memberId 8을 사용
            command="빅뱅 노래 추천해줘"  # 서버에 전달할 명령
        )

        # 서버에 요청을 보내고 응답 받기
        response = stub.GetLangchainAgentRecommendation(request)

        # 응답 처리
        print("Received Langchain Recommendation:")
        for item in response.searchResult:
            print(f"Song Info ID: {str(item.songInfoId)}\n")
            print(f"Song Name: {item.reason}\n")
            print(f"Singer Name: {item.singerName}\n")
            print("-" * 20)
            print("\n")

if __name__ == "__main__":
    run()