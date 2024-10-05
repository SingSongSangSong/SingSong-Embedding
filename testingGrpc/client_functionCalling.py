import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import grpc
from proto.functionCallingRecommend.functionCallingRecommend_pb2 import FunctionCallingRequest, FunctionCallingResponse
from proto.functionCallingRecommend.functionCallingRecommend_pb2_grpc import functionCallingRecommendStub

def run():
    # gRPC 서버에 연결
    with grpc.insecure_channel('localhost:50051') as channel:
        # LangchainRecommendStub을 사용해 서버와 통신
        stub = functionCallingRecommendStub(channel)

        # LangchainRequest 생성
        request = FunctionCallingRequest(
            memberId=8,  # 예시로 memberId 8을 사용
            command="윤하의 사건의지평선과 비슷한 노래 추천해줘"  # 서버에 전달할 명령
        )

        # 서버에 요청을 보내고 응답 받기
        response = stub.GetFunctionCallingRecommendation(request)

        # 응답 처리
        print("Received Langchain Recommendation:")
        for item in response.songInfoId:
            print(f"Song Info ID: {str(item)}\n")
            

if __name__ == "__main__":
    run()