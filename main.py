import grpc
from concurrent import futures
from service.userProfileService import UserProfileService
from service.milvusInsertService import MilvusInsertService
import logging
from service.userProfileServiceGrpc import UserProfileServiceGrpc
from service.langchainServiceGrpc import LangChainServiceGrpc
from proto.userProfileRecommend.userProfileRecommend_pb2_grpc import add_UserProfileServicer_to_server
from proto.langchainRecommend.langchainRecommend_pb2_grpc import add_LangchainRecommendServicer_to_server
from service.hotTrendingService import HotTrendingService
from db.dbConfig import DatabaseConfig
from apscheduler.schedulers.background import BackgroundScheduler 

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed output
logger = logging.getLogger(__name__)

# gRPC 서버를 실행하는 함수
def serve_grpc():
    logger.info("Starting gRPC server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    add_UserProfileServicer_to_server(UserProfileServiceGrpc(user_profile_service), server)
    add_LangchainRecommendServicer_to_server(LangChainServiceGrpc(), server)

    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("gRPC server started on port 50051")

    try:
        server.wait_for_termination()  # gRPC 서버 종료 대기
    except (KeyboardInterrupt, SystemExit):
        background_scheduler.shutdown()
        logger.info("Scheduler and gRPC server stopped.")
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
    hot_trending_service = HotTrendingService() # db config, redis config
    hot_trending_service.v2_init()

    background_scheduler = BackgroundScheduler(timezone='Asia/Seoul')
    background_scheduler.add_job(hot_trending_service.v2_scheduler, 'cron', minute='50', id='hot_trending_scheduler')
    background_scheduler.add_job(job, 'cron', minute='55', id='user_profile_scheduler')
    background_scheduler.start()
    logger.info("Background scheduler started")

    # gRPC 서버 실행
    grpc_server = serve_grpc()