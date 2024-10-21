import grpc
import logging
import asyncio
import signal
from ddtrace import tracer
from apscheduler.schedulers.asyncio import AsyncIOScheduler  # 비동기 스케줄러
from service.userProfileService import UserProfileService
from service.userProfileServiceGrpc import UserProfileServiceGrpc
from service.functionCallingServiceGrpc import FunctionCallingServiceGrpc
from service.funtionCallingWithTypes import FunctionCallingWithTypesServiceGrpc
from proto.userProfileRecommend.userProfileRecommend_pb2_grpc import add_UserProfileServicer_to_server
from proto.functionCallingRecommend.functionCallingRecommend_pb2_grpc import add_functionCallingRecommendServicer_to_server
from proto.functionCallingWithTypes.functionCallingWithTypes_pb2_grpc import add_FunctionCallingWithTypesRecommendServicer_to_server
from service.hotTrendingService import HotTrendingService
from service.tjCrawlingService import TJCrawlingService

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 전역 변수 초기화
user_profile_service = UserProfileService()
hot_trending_service = HotTrendingService()
crawling_service = TJCrawlingService()

# 전역 스케줄러와 서버 참조
scheduler = None
grpc_server = None

# 동기 작업을 비동기적으로 실행
async def user_profile_job():
    logger.info("Running user profile creation process.")
    await user_profile_service.run()
    logger.info("User profile creation process finished.")

async def hot_trending_job():
    logger.info("Running hot trending job.")
    await hot_trending_service.v2_scheduler()
    logger.info("Hot trending job completed.")

# gRPC 서버 실행
async def serve_grpc():
    global grpc_server
    grpc_server = grpc.aio.server()  # 비동기 gRPC 서버 생성

    # 서비스 추가
    add_UserProfileServicer_to_server(UserProfileServiceGrpc(user_profile_service), grpc_server)
    add_functionCallingRecommendServicer_to_server(FunctionCallingServiceGrpc(), grpc_server)
    add_FunctionCallingWithTypesRecommendServicer_to_server(FunctionCallingWithTypesServiceGrpc(), grpc_server)

    grpc_server.add_insecure_port('[::]:50051')
    await grpc_server.start()
    logger.info("gRPC server started on port 50051")

# Graceful Shutdown 핸들러
async def shutdown():
    global scheduler, grpc_server

    logger.info("Shutting down gracefully...")

    # 스케줄러 중지
    if scheduler:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped.")

    # gRPC 서버 중지
    if grpc_server:
        await grpc_server.stop(5)  # 최대 5초 동안 종료 대기
        logger.info("gRPC server stopped.")

    logger.info("Shutdown completed.")

# 신호 처리기 등록
def register_signal_handlers(loop):
    for signal_name in ('SIGINT', 'SIGTERM'):
        loop.add_signal_handler(
            getattr(signal, signal_name),
            lambda: asyncio.ensure_future(shutdown())
        )

# 비동기 메인 함수
async def main():
    global scheduler

    tracer.configure(hostname='datadog-agent', port=8126, https=False)

    # HotTrendingService 및 크롤링 서비스 초기화
    await hot_trending_service.v2_init()

    # 비동기 스케줄러 설정 및 시작
    scheduler = AsyncIOScheduler(timezone='Asia/Seoul')
    scheduler.add_job(hot_trending_job, 'cron', minute='50', id='hot_trending_scheduler')
    scheduler.add_job(user_profile_job, 'cron', hour='03', minute='55', id='user_profile_scheduler')
    scheduler.add_job(crawling_service.crawl_and_save_new_songs, 'cron', hour='11', minute='0', id='daily_new_song_scheduler')
    scheduler.start()
    logger.info("Background scheduler started")

    # gRPC 서버 시작
    await serve_grpc()

    # gRPC 서버 종료 대기
    await grpc_server.wait_for_termination()

if __name__ == "__main__":
    # 이벤트 루프 가져오기
    loop = asyncio.get_event_loop()

    # 신호 처리기 등록
    register_signal_handlers(loop)

    # 비동기 메인 함수 실행
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("Event loop closed.")