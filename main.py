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
from service.tjCrawlingService import TJCrawlingService
from db.dbConfig import DatabaseConfig
from apscheduler.schedulers.background import BackgroundScheduler
import threading
from flask import Flask, jsonify  # Flask 임포트

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more detailed output
logger = logging.getLogger(__name__)

# Flask 앱 생성
app = Flask(__name__)
grpc_server_running = False  # gRPC 서버 상태 추적 변수

# gRPC 서버를 실행하는 함수
def serve_grpc():
    global grpc_server_running
    logger.info("Starting gRPC server...")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    add_UserProfileServicer_to_server(UserProfileServiceGrpc(user_profile_service), server)
    add_LangchainRecommendServicer_to_server(LangChainServiceGrpc(), server)

    server.add_insecure_port('[::]:50051')
    server.start()
    grpc_server_running = True  # gRPC 서버가 실행 중임을 나타냄
    logger.info("gRPC server started on port 50051")

    try:
        server.wait_for_termination()  # gRPC 서버 종료 대기
    except (KeyboardInterrupt, SystemExit):
        background_scheduler.shutdown()
        logger.info("Scheduler and gRPC server stopped.")
        grpc_server_running = False  # gRPC 서버가 중지되었음을 표시

# Flask Health Check 엔드포인트
@app.route("/health", methods=["GET"])
def health_check():
    if grpc_server_running:
        return jsonify({"status": "gRPC server running"}), 200
    else:
        return jsonify({"status": "gRPC server not running"}), 503

# Flask 서버를 실행하는 함수
def run_flask():
    app.run(host="0.0.0.0", port=8080)

# Milvus insert 실행 함수
def run_milvus_insert_script():
    logger.info("Running Milvus insert process...")
    
    # MilvusInsertService 인스턴스 생성
    milvus_service = MilvusInsertService(collection_name="singsongsangsong_22286")
    
    # 데이터 삽입 프로세스 실행
    milvus_service.run(song_info_path="dataframe/song_info.csv", data_path="dataframe/with_ssss_22286_updated5.csv")

    logger.info("Milvus insert process completed successfully.")

# 매 45분마다 실행할 작업 (UserProfileService의 run 메서드 호출)
def job():
    logger.info("Running user profile creation process.")
    user_profile_service.run()  # user_profile_service의 run 메서드를 호출
    logger.info("User profile creation process finished.")

if __name__ == "__main__":
    # Flask 서버를 백그라운드 스레드에서 실행
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.start()

    # Milvus insert 작업을 메인 스레드에서 실행
    logger.info("Running Milvus insert process...")
    run_milvus_insert_script()  # 스레드로 실행하지 않고, 순차적으로 실행
    logger.info("Milvus insert process completed. Starting gRPC server...")

    # UserProfileService 인스턴스 생성
    user_profile_service = UserProfileService()
    user_profile_service.create_user_profile_collection()
    user_profile_service.create_gender_profiles()
    
    hot_trending_service = HotTrendingService()  # db config, redis config
    hot_trending_service.v2_init()

    crawling_service = TJCrawlingService()

    # Background scheduler 시작
    background_scheduler = BackgroundScheduler(timezone='Asia/Seoul')
    background_scheduler.add_job(hot_trending_service.v2_scheduler, 'cron', minute='50', id='hot_trending_scheduler')
    background_scheduler.add_job(job, 'cron', minute='55', id='user_profile_scheduler')
    background_scheduler.add_job(crawling_service.crawl_and_save_new_songs, 'cron', day='1', hour='3', minute='0', id='monthly_new_song_scheduler')
    background_scheduler.start()
    logger.info("Background scheduler started")

    # Milvus insert 작업이 완료된 후 gRPC 서버를 백그라운드 스레드에서 실행
    grpc_thread = threading.Thread(target=serve_grpc)
    grpc_thread.start()

    # gRPC 서버 종료 대기
    grpc_thread.join()

    # Flask 서버 종료 대기
    flask_thread.join()