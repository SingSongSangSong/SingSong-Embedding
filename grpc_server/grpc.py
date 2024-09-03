import logging
import grpc
from concurrent import futures
from google.protobuf.empty_pb2 import Empty
from proto.embedding_pb2 import ProfileResponse
from proto.embedding_pb2_grpc import UserProfileServicer, add_UserProfileServicer_to_server
from service.embedding_service import EmbeddingService
from sqlalchemy.orm import Session
from db.dbConfig import get_db

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserProfileServicerImpl(UserProfileServicer):
    def CreateUserProfile(self, request, context):
        # EmbeddingService와 DB 세션을 초기화
        service = EmbeddingService(model_name='sentence-transformers/all-MiniLM-L6-v2')
        db = next(get_db())  # get_db는 generator이므로 next()로 세션을 얻습니다.

        try:
            # 기본 성별별 유저 프로파일 생성
            service.create_gender_based_profiles(db)

            # 최근 활성 멤버 가져오기
            recent_active_members = service.fetch_recent_active_members(db)
            member_count = recent_active_members.shape[0] if recent_active_members is not None else 0
            logger.info(f"Fetched {member_count} recent active members")

            if recent_active_members is not None and not recent_active_members.empty:
                # 각 멤버의 행동 데이터 가져오기
                member_action_data = service.fetch_member_action_data(db, recent_active_members['member_id'].tolist())
                action_data_count = member_action_data.shape[0] if member_action_data is not None else 0
                logger.info(f"Fetched {action_data_count} member action records")

                # 전체 데이터 로드
                total_data = service.load_total_data()
                total_data_count = total_data.shape[0] if total_data is not None else 0
                logger.info(f"Loaded {total_data_count} total data records")

                if member_action_data is not None and total_data is not None:
                    # 데이터 병합
                    merged_data = service.merge_data(member_action_data, total_data)
                    merged_data_count = merged_data.shape[0] if merged_data is not None else 0
                    logger.info(f"Merged into {merged_data_count} records")

                    if merged_data is not None:
                        weights = {'CLICK': 1, 'KEEP': 5}
                        grouped = merged_data.groupby('member_id')

                        for member_id, group in grouped:
                            genre_pref, year_pref, country_pref, singer_type_pref, ssss_pref, max_pitch_pref = service.calculate_user_preferences(group, weights)

                            # 각 사용자 선호도 계산
                            if genre_pref is not None:
                                user_preferences_sentence = service.create_user_preference_sentence(
                                    genre_pref, year_pref, country_pref, singer_type_pref, ssss_pref, max_pitch_pref
                                )

                                # 선호도 임베딩 생성
                                user_embedding = service.embed_user_preferences(user_preferences_sentence)
                                logger.info(f"Generated embedding for member_id {member_id}")

                                if user_embedding is not None:
                                    # 사용자 프로필 업데이트 또는 삽입
                                    service.update_or_insert_user_profile(db, member_id, user_embedding)
                                    logger.info(f"Updated or inserted profile for member_id {member_id}")
        finally:
            db.close()  # 세션 닫기
        
        return ProfileResponse(status="200")

def serve_grpc():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_UserProfileServicer_to_server(UserProfileServicerImpl(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("gRPC server started on port 50051")
    server.wait_for_termination()