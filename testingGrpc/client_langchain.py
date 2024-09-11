import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import grpc
from proto.langchainRecommend.langchainRecommend_pb2_grpc import LangchainRecommendStub
from proto.langchainRecommend.langchainRecommend_pb2 import LangchainRequest

def run():
    # gRPC 서버에 연결
    with grpc.insecure_channel('localhost:50051') as channel:
        # LangchainRecommendStub을 사용해 서버와 통신
        stub = LangchainRecommendStub(channel)

        # LangchainRequest 생성
        request = LangchainRequest(
            memberId=8,  # 예시로 memberId 8을 사용
            command="버즈의 가시와 비슷한 노래를 추천해줘"  # 서버에 전달할 명령
        )

        # 서버에 요청을 보내고 응답 받기
        response = stub.GetLangchainRecommendation(request)

        # 응답 처리
        print("Received Langchain Recommendation:")
        for item in response.similarItems:
            print(f"Song Info ID: {item.songInfoId}")
            print(f"Song Name: {item.songName}")
            print(f"Singer Name: {item.singerName}")
            print(f"Is MR: {item.isMr}")
            print(f"SSSS: {item.ssss}")
            print(f"Audio File URL: {item.audioFileUrl}")
            print(f"Album: {item.album}")
            print(f"Song Number: {item.songNumber}")
            print(f"Similarity Score: {item.similarityScore}")
            print("-" * 20)

if __name__ == "__main__":
    run()