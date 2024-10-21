import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import grpc
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2 import FunctionCallingWithTypesRequest, FunctionCallingWithTypesResponse
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2_grpc import FunctionCallingWithTypesRecommendStub
import time

def run():
    # gRPC 서버에 연결
    with grpc.insecure_channel('localhost:50051') as channel:
        # LangchainRecommendStub을 사용해 서버와 통신
        stub = FunctionCallingWithTypesRecommendStub(channel)

        # 테스트할 프롬프트 리스트
        prompts = [
            "3옥타브 레 남자노래 추천해줘"            
        ]

        prompt2 = [
            "30대 남자가 부르기 어려운 노래 추천해줘",
            "2000년도부터 2010년 사이에 나온 노래 추천해줘",
            "최고음 3옥타브레 인 노래 추천해줘",
            "한국인 말고 일본인이 부른 j-pop 추천해줘",
            "해외 가수가 부른 R&B 추천",
            "R&B 팝송 추천"
            "정승환 눈사람 같은 노래",
            "누구나 아는 케이팝 추천해줘",
            "유명한 발라드",
            "감성 충만한 발라드 추천해줘, 노래방에서 부르고 싶어",
            "2020년대 유명한 발라드",
            "쿨의 애상 같은 노래 찾아줘",
            "저음도 잘 부르는 편인데, 저음 매력적인 노래 뭐 있어?",
            "저음인 사람들이 쉽게 부를수있는노래 찾아줘",
            "혁오의 TOMBOY 같은 노래 추천해줘",
            "노래 잘불러보이는 곡 추천해줘",
            "혁오 tomboy와 비슷한 노래 추천",
            "디즈니 노래 추천",
            "혼자 노래방 갈 때 부르기 좋은 노래 있을까?",
            "여자 고음 노래",
            "오늘 헤어졌는데 부를만한 노래 추천좀..",
            "최근 발라드중 인기가 많았던 노래를 얘기해줘",
            "락발라드인데 옛날느낌나는 쉬운 노래 추천해줘",
            "유명한 락발라드중 부르기쉬운 노래 추천해줘",
            "90년도 부르기 쉬운 김경호 노래 추천해줘",
            "발라드 추천해줘",
            "노래방에서 부르기 쉬운 발라드 추천해줘",
            "혼술하고싶은 밤 같은 노래 추천해줘",
            "썸 탈 때 부르기 좋은 노래",
            "발라드인데 주저하는 연인들을 위하여랑 비슷한느낌",
            "친구들이랑 신나게 부를 수 있는 노래 알려줘",
            "최근 노래중에 신나는 노래",
            "애절한 감성 가득한 노래 추천해줘, 부르고 싶어",
            "발라드 노래 추천해줘",
            "노래방에서 분위기 띄우기 좋은 신나는 곡 추천해줘",
            "사랑 노래",
            "너가 돌아오지 않을거란걸알아",
            "BTS의 핸드폰 좀 꺼줄래 같은 노래 추천해줘",
            "요즘 노래방에서 유행하는 핫한 곡 추천해줘",
            "노래방에서 친구들이랑 떼창하기 좋은 곡 추천해줘",
            "애절한 감성 가득한 노래 추천해줘, 부르고 싶어",
            "오늘 헤어졌는데 부를만한 노래 추천좀..",
            "우울할때 들을수있는노래 추천해줘",
            "고등래퍼 노래 추천",
            "커넥션 고백 같은 거 추천",
            "부르기 쉬운 노래",
            "애절한 감성 가득한 노래 추천해줘, 부르고 싶어",
            "음치도 부르기 쉬운 노래",
            "군대 동기랑 외출",
            "친구랑 신나는 노래",
            "쇼미더머니 추천 노래",
            "감성 충만한 발라드 추천해줘, 노래방에서 부르고 싶어",
            "2020년대 노래 추천해줘",
            "2024년에 나온 노래 추천해줘",
            "분위기 띄울 수 있고 친구들도 다 알만한 국내 힙합곡 추천해줘",
            "음.. 난 2007년생인 거 감안해서 친구들이 잘 알만한 국내 힙합 추천해줘",
            "2020년 기준 최대 2년 안팎으로 분위기 띄우기 좋은 국내 힙합 추천 좀",
            "중저음 여자가 부를 만한 인디/R&B 추천",
            "2023 기준 국내 힙합 곡 추천",
            "음색이 좋으면 잘 부르게 들리는 딘의 Instagram 같은 곡 추천해줘",
            "가성 연습할 때 부르는 노래",
            "저음도 잘 부르는 편인데, 저음 매력적인 노래 뭐 있어?",
            "09년생인데 친구들이 알만한 랩 알려줘",
            "요즘 싱잉랩 추천좀",
            "여자가 부를만한 싱잉랩 알려줘",
            "음이 높지 않은 감성있는 여자 노래"
        ]


        # 각 프롬프트에 대해 서버에 요청 보내기
        for prompt in prompts:
            print(f"Sending request with prompt: {prompt}")
            
            # LangchainRequest 생성
            request = FunctionCallingWithTypesRequest(
                memberId=8,  # 예시로 memberId 8을 사용
                gender='MALE',
                year='2020',
                command=prompt  # 서버에 전달할 명령
            )

            # 서버에 요청을 보내고 응답 받기
            response = stub.GetFunctionCallingWithTypesRecommendation(request)

            # 응답 처리
            print("Received Langchain Recommendation for prompt:")
            print(f"Prompt: {prompt}")
            for item in response.songInfos:
                print(f"Song Info ID: {str(item)}\n")

            # 테스트 사이에 잠시 대기 (필요에 따라 시간 조절)
            time.sleep(1)

if __name__ == "__main__":
    run()