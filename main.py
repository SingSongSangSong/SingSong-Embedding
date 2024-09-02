# 이 코드가 현재의 main.py
from grpc_server.grpc import serve_grpc  # gRPC 서버를 실행하는 함수를 import

if __name__ == "__main__":
    serve_grpc()  # gRPC 서버만 실행
