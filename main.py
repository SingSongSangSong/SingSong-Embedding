import schedule
import time
import grpc
import subprocess
from concurrent import futures
from service.userProfileService import UserProfileService
from service.milvusInsertService import MilvusInsertService
import logging
import threading
from service.userProfileServiceGrpc import UserProfileServiceGrpc
from proto.userProfileRecommend.userProfileRecommend_pb2_grpc import add_UserProfileServicer_to_server


# 로깅 설정
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed output
logger = logging.getLogger(__name__)

# gRPC 서버를 실행하는 함수
def serve_grpc():
    logger.info("Starting gRPC server in a separate thread...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    add_UserProfileServicer_to_server(UserProfileServiceGrpc(user_profile_service), server)

    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC server started on port 50051")
    server.wait_for_termination()

# 매 45분마다 실행할 작업 (UserProfileService의 run 메서드 호출)
def job():
    logger.info("Running user profile creation process.")
    user_profile_service.run()  # user_profile_service의 run 메서드를 호출
    logger.info("User profile creation process finished.")

# Milvus insert 실행 함수
def run_milvus_insert_script():
    logger.info("Running Milvus insert process...")
    
    # MilvusInsertService 인스턴스 생성
    milvus_service = MilvusInsertService(collection_name="singsongsangsong_22286")
    
    # 데이터 삽입 프로세스 실행
    milvus_service.run(song_info_path="dataframe/song_info.csv", data_path="dataframe/with_ssss_22286_updated5.csv")

    logger.info("Milvus insert process completed successfully.")

if __name__ == "__main__":
    # Milvus insert script 실행
    run_milvus_insert_script()

    # UserProfileService 인스턴스 생성
    user_profile_service = UserProfileService()
    user_profile_service.create_user_profile_collection()
    user_profile_service.create_gender_profiles()

    # gRPC 서버를 별도의 스레드에서 실행
    grpc_thread = threading.Thread(target=serve_grpc)
    grpc_thread.daemon = True
    grpc_thread.start()

    # 매 1440분(24시간)마다 user_profile_service의 run 메서드를 실행하는 작업 스케줄 등록
    schedule.every(1440).minutes.do(job)

    logger.info("Scheduler started, running every 1440 minutes.")

    # 스케줄러를 실행하여 무한 루프를 돌면서 주기적으로 작업 실행
    while True:
        logger.debug("Waiting for scheduled jobs...")
        schedule.run_pending()
        time.sleep(1)