import grpc
from google.protobuf.empty_pb2 import Empty  # 빈 메시지를 위한 임포트

def health_check():
    # gRPC 서버의 주소와 포트 (localhost:50051로 예시)
    channel = grpc.insecure_channel('localhost:50051')

    try:
        # 빈 요청을 직렬화
        request = Empty()
        serialized_request = request.SerializeToString()  # 직렬화된 바이트 형식

        # /AWS.ALB/healthcheck 호출
        response = channel.unary_unary('/AWS.ALB/healthcheck').future(serialized_request, timeout=5)
        print(f"Health Check Response: {response.result()}")
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.UNIMPLEMENTED:
            print("Received expected response: 12 (Unimplemented)")
        else:
            print(f"Unexpected error: {e.code()} - {e.details()}")

if __name__ == "__main__":
    health_check()