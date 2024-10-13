import grpc
from proto.userProfileRecommend.userProfileRecommend_pb2 import ProfileRequest
from proto.userProfileRecommend.userProfileRecommend_pb2_grpc import UserProfileStub

# 서버에 gRPC 요청을 보내는 클라이언트 코드
def run():
    # 서버 연결
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = UserProfileStub(channel)
        
        # ProfileRequest 요청
        request = ProfileRequest(memberId=1, page=20, gender="MALE")
        response = stub.CreateUserProfile(request)
        
        # 응답 출력
        print("Received response:")
        for item in response.similar_items:
            print(f"Song ID: {item.song_info_id}, Song Name: {item.song_name}, Artist: {item.artist_name}")

if __name__ == "__main__":
    run()