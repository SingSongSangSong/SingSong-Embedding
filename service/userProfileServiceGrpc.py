import grpc
from concurrent import futures
import logging
import numpy as np
from proto.userProfileRecommend.userProfileRecommend_pb2_grpc import UserProfileServicer
from proto.userProfileRecommend.userProfileRecommend_pb2 import ProfileRequest, ProfileResponse, SimilarItem
from pymilvus import Collection, connections
import os

logger = logging.getLogger(__name__)

class UserProfileServiceGrpc(UserProfileServicer):
    def __init__(self, user_profile_service):
        self.user_profile_service = user_profile_service
        # Milvus 연결
        connections.connect(host=os.getenv("MILVUS_HOST", "milvus-standalone"), port="19530")
        self.song_collection = Collection("singsongsangsong_22286")
        self.profile_collection = Collection("user_profile")

    # memberId에 대한 유저 프로파일 조회
    def get_user_profile(self, member_id):
        expr = f"member_id == {member_id}"
        profiles = self.profile_collection.query(expr=expr, output_fields=["profile_vector"])
        if profiles:
            return profiles[0]['profile_vector']  # vector를 반환
        return None  # 프로파일이 없을 경우 None

    # 유사도 기반 추천 로직
    def recommend_similar_songs(self, user_vector, top_k=20):
        logger.info("Recommending songs based on user profile vector.")
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
        user_embedding = np.array(user_vector, dtype=np.float32).reshape(1, -1)
        
        # Milvus 검색
        search_results = self.song_collection.search(
            data=user_embedding,
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr="MR == False",  # MR이 False인 항목만 검색
            output_fields=["song_info_id", "song_name", "artist_name", "MR", "ssss", "audio_file_url", "album", "song_number"]
        )
        return search_results

    # gRPC CreateUserProfile 메서드
    def CreateUserProfile(self, request, context):
        logger.info(f"Received gRPC request to create user profile for memberId: {request.memberId}, page: {request.page}, gender: {request.gender}")
        try:
            # 유저 프로파일 조회
            user_vector = self.get_user_profile(request.memberId)

            if not user_vector:
                # 멤버 ID에 해당하는 프로필이 없을 경우, gender 기반 기본 프로필 조회
                logger.warning(f"No user profile found for memberId: {request.memberId}. Looking for gender-based default profile.")

                # gender가 male이면 member_id=0, female이면 member_id=-1
                if request.gender.upper() == 'MALE':
                    logger.info("Fetching default profile for male users.")
                    user_vector = self.get_user_profile(0)  # 남성 기본 프로파일
                elif request.gender.upper() == 'FEMALE':
                    logger.info("Fetching default profile for female users.")
                    user_vector = self.get_user_profile(-1)  # 여성 기본 프로파일
                elif request.gender.upper() == 'UNKNOWN':
                    logger.info("Fetching default profile for Unkown users.")
                    user_vector = self.get_user_profile(-2)  # Unknown 기본 프로파일

            # 기본 프로파일도 없을 경우 빈 응답 반환
            if not user_vector:
                logger.warning(f"No profile found for memberId: {request.memberId} or gender: {request.gender}. Returning empty response.")
                return ProfileResponse(similarItems=[])

            # 유사도 기반 추천
            logger.info(f"Found user profile for memberId: {request.memberId} or gender: {request.gender}")
            search_results = self.recommend_similar_songs(user_vector, top_k=request.page)

            # 결과를 gRPC 응답으로 변환
            similar_items = []
            for result in search_results:
                for hit in result:
                    logger.info(hit)
                    similar_items.append(SimilarItem(
                        songInfoId=hit.id,  # song_info_id
                        songName=hit.entity.song_name,  # 노래 제목
                        singerName=hit.entity.artist_name,  # 아티스트 이름
                        isMr=hit.entity.MR,  # MR 여부
                        ssss=hit.entity.ssss,  # 추가 메타데이터 필드
                        audioFileUrl=hit.entity.audio_file_url,  # 오디오 파일 URL
                        album=hit.entity.album,  # 앨범 이름
                        songNumber=hit.entity.song_number,  # 곡 번호
                        similarityScore=hit.distance  # 유사도 점수
                    ))

            return ProfileResponse(similarItems=similar_items)

        except Exception as e:
            logger.error(f"Error during user profile creation: {e}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Internal server error")
            return ProfileResponse(similarItems=[])