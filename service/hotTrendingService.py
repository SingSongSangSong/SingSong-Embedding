import json
import logging
import os

import pymysql
import redis
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9 이상
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HotTrendingService:
    def __init__(self):
        self.db_host = os.getenv('DB_HOST')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_database = os.getenv('DB_DATABASE')
        self.db_port = 3306
        
        self.redis_host = os.getenv('REDIS_HOST')
        self.redis_port = 6379

    def setup_db_config(self):
        try:
            db = pymysql.connect(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                database=self.db_database,
                port=self.db_port,
                charset='utf8mb4',
                cursorclass=pymysql.cursors.DictCursor
            )
        except pymysql.MySQLError as e:
            logger.error(f"MySQL 연결 실패: {e}")
            raise

        return db
    
    def setup_redis_config(self):
        try:
            rdb = redis.Redis(
                host=self.redis_host,
                port=self.redis_port
            )
            rdb.ping()
            logger.info("Redis 연결 성공")
        except redis.ConnectionError as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise

        logger.info("MySQL 및 Redis 연결 성공")
        return rdb

    # V1: 남성 전체, 여성 전체 - live 없음
    def v1_scheduler(self):
        try:
            db = self.setup_db_config()
            cursor = db.cursor()

            cursor.execute("""
                WITH scored_songs AS (
                    SELECT
                        subquery.song_info_id,
                        SUM(subquery.action_score) AS total_score
                    FROM (
                        SELECT DISTINCT
                            ma.member_id,
                            ma.song_info_id,
                            ma.action_type,
                            ma.action_score,
                            ma.gender
                        FROM member_action as ma
                        WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 1 MONTH)
                        AND ma.gender = 'MALE'
                    ) AS subquery
                    GROUP BY subquery.song_info_id
                )
                SELECT
                    RANK() OVER (ORDER BY ss.total_score DESC) AS ranking,
                    ss.song_info_id,
                    ss.total_score,
                    s.song_name,
                    s.artist_name,
                    s.song_number,
                    s.is_mr
                FROM scored_songs ss
                JOIN song_info s ON ss.song_info_id = s.song_info_id
                ORDER BY ss.total_score DESC
                LIMIT 20;
            """)
            male_results = cursor.fetchall()

            cursor.execute("""
                WITH scored_songs AS (
                    SELECT
                        subquery.song_info_id,
                        SUM(subquery.action_score) AS total_score
                    FROM (
                        SELECT DISTINCT
                            ma.member_id,
                            ma.song_info_id,
                            ma.action_type,
                            ma.action_score,
                            ma.gender
                        FROM member_action as ma
                        WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 1 MONTH)
                        AND ma.gender = 'FEMALE'
                    ) AS subquery
                    GROUP BY subquery.song_info_id
                )
                SELECT
                    RANK() OVER (ORDER BY ss.total_score DESC) AS ranking,
                    ss.song_info_id,
                    ss.total_score,
                    s.song_name,
                    s.artist_name,
                    s.song_number,
                    s.is_mr
                FROM scored_songs ss
                JOIN song_info s ON ss.song_info_id = s.song_info_id
                ORDER BY ss.total_score DESC
                LIMIT 20;
            """)
            female_results = cursor.fetchall()
            db.close()

            rdb = self.setup_redis_config()
            seoul_tz = ZoneInfo('Asia/Seoul')
            now = datetime.now(seoul_tz)

            one_hour_later = now + timedelta(hours=1)
            formatted_string_for_one_hour_later = one_hour_later.strftime("%Y-%m-%d-%H-Hot_Trend")

            try:
                formatted_string_for_current_time = now.strftime("%Y-%m-%d-%H-Hot_Trend")

                male_exists = rdb.exists(formatted_string_for_current_time + "_MALE")
                female_exists = rdb.exists(formatted_string_for_current_time + "_FEMALE")

                if male_exists:
                    current_male_data = rdb.get(formatted_string_for_current_time + "_MALE")
                    male_json_data = json.loads(current_male_data)
                else:
                    male_json_data = []

                if female_exists:
                    current_female_data = rdb.get(formatted_string_for_current_time + "_FEMALE")
                    female_json_data = []
                    female_json_data = json.loads(current_female_data)
                else:
                    female_json_data = []

                previous_male_ids = {item["song_info_id"]: item["ranking"] for item in male_json_data}
                previous_female_ids = {item["song_info_id"]: item["ranking"] for item in female_json_data}

                for male_value in male_results:
                    song_info_id = male_value["song_info_id"]
                    if song_info_id not in previous_male_ids:
                        male_value["new"] = "new"
                        male_value["ranking_change"] = 0
                    else:
                        male_value["new"] = "old"
                        male_value["ranking_change"] = previous_male_ids[song_info_id] - male_value["ranking"]

                for female_value in female_results:
                    song_info_id = female_value["song_info_id"]
                    if song_info_id not in previous_female_ids:
                        female_value["new"] = "new"
                        female_value["ranking_change"] = 0
                    else:
                        female_value["new"] = "old"
                        female_value["ranking_change"] = previous_female_ids[song_info_id] - female_value["ranking"]

            except Exception as e:
                logger.error(f"현재 데이터를 Redis에서 가져오는 데 실패했습니다: {e}")
                raise

            new_formatted_string_for_male = formatted_string_for_one_hour_later + "_MALE"
            rdb.set(new_formatted_string_for_male, json.dumps(male_results))
            rdb.expire(new_formatted_string_for_male, 4800)

            new_formatted_string_for_female = formatted_string_for_one_hour_later + "_FEMALE"
            rdb.set(new_formatted_string_for_female, json.dumps(female_results))
            rdb.expire(new_formatted_string_for_female, 4800)

        except Exception as e:
            logger.error(f"실행 중 오류 발생: {e}")

    # V2 - db에 live 추가하면 변경 필요
    # 성별미정 전체, 10대, 20대, 30대, 40대 이상
    # 남성 전체, 10대, 20대, 30대, 40대 이상
    # 여성 전체, 10대, 20대, 30대, 40대 이상
    def v2_scheduler(self):
        try: 
            db = self.setup_db_config()
            cursor = db.cursor()
            cursor.execute("""     
                WITH scored_songs AS (
                    SELECT
                        ma.song_info_id,
                        SUM(ma.action_score) AS total_score,
                        ma.gender,
                        CASE
                            WHEN ma.birthyear = 0 THEN 'unknown'
                            WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 10 AND 19 THEN '10'
                            WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 20 AND 29 THEN '20'
                            WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 30 AND 39 THEN '30'
                            WHEN YEAR(CURDATE()) - ma.birthyear + 1 > 39 THEN '40+'
                        END AS age_group
                    FROM member_action as ma
                    WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 1 MONTH)
                    GROUP BY ma.song_info_id, ma.gender, age_group
                )
                SELECT
                    ss.song_info_id,
                    ss.total_score,
                    s.song_name,
                    s.artist_name,
                    s.song_number,
                    s.is_mr,
                    s.album,
                    ss.gender,
                    ss.age_group
                FROM scored_songs ss
                JOIN song_info s ON ss.song_info_id = s.song_info_id
            """)
            results = cursor.fetchall()
            db.close()

            # is_live 추가 - 임시
            for _, item in enumerate(results):
                item["is_live"] = 0

            rdb = self.setup_redis_config()

            # Redis에서 이전 데이터 가져오기
            seoul_tz = ZoneInfo('Asia/Seoul')
            now = datetime.now(seoul_tz)
            formatted_string_for_current_time = now.strftime("%Y-%m-%d-%H-Hot_Trend")

            previous_data = {"male": {}, "female": {}, "mixed": {}}
            for gender_key in ["MALE", "FEMALE", "MIXED"]:
                for age_group in ["ALL", "10", "20", "30", "40+"]:
                    key = f"{formatted_string_for_current_time}_{gender_key}_{age_group}"
                    if rdb.exists(key):
                        current_data = rdb.get(key)
                        previous_data[gender_key.lower()][age_group] = json.loads(current_data)
                    else:
                        previous_data[gender_key.lower()][age_group] = []

            # 성별 및 나이대별로 데이터를 분류하기 위한 저장소
            male_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
            female_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
            mixed_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}

            # 데이터를 성별 및 나이대에 맞게 분류
            for row in results:
                gender = row['gender']
                age_group = row['age_group']
                mixed_data['ALL'].append(row)
                mixed_data[age_group].append(row)
                if gender == 'MALE':
                    male_data['ALL'].append(row)
                    male_data[age_group].append(row)
                elif gender == 'FEMALE':
                    female_data['ALL'].append(row)
                    female_data[age_group].append(row)

            # 각 케이스에 대해 상위 20개의 노래만 자르고 ranking과 ranking_change 추가
            for age_group in male_data.keys():
                male_data[age_group] = self.add_ranking_info(self.get_top_20_by_score(male_data[age_group]), previous_data["male"][age_group])
                female_data[age_group] = self.add_ranking_info(self.get_top_20_by_score(female_data[age_group]), previous_data["female"][age_group])
                mixed_data[age_group] = self.add_ranking_info(self.get_top_20_by_score(mixed_data[age_group]), previous_data["mixed"][age_group])

            # Redis에 저장할 키 생성
            one_hour_later = now + timedelta(hours=1)
            formatted_string_for_one_hour_later = one_hour_later.strftime("%Y-%m-%d-%H-Hot_Trend")

            # Redis에 저장 (남성, 여성, 성별미정)
            for age_group in male_data.keys():
                male_key = f"{formatted_string_for_one_hour_later}_MALE_{age_group}"
                female_key = f"{formatted_string_for_one_hour_later}_FEMALE_{age_group}"
                combined_key = f"{formatted_string_for_one_hour_later}_MIXED_{age_group}"

                rdb.set(male_key, json.dumps(male_data[age_group]))
                rdb.expire(male_key, 4800)

                rdb.set(female_key, json.dumps(female_data[age_group]))
                rdb.expire(female_key, 4800)

                rdb.set(combined_key, json.dumps(mixed_data[age_group]))
                rdb.expire(combined_key, 4800)

            rdb.close()
            logger.info("hot trending 갱신 성공")

        except Exception as e:
            logger.exception("hot trending 갱신 중 오류 발생")
            

    def get_top_20_by_score(self, data_list):
        # total_score 기준으로 내림차순 정렬 후 상위 20개 추출
        return sorted(data_list, key=lambda x: x['total_score'], reverse=True)[:20]
    
    def add_ranking_info(self, current_data, previous_data):
        previous_ranking = {item["song_info_id"]: item["ranking"] for item in previous_data}

        for idx, item in enumerate(current_data):
            current_ranking = idx + 1  # 1위부터 시작
            item["ranking"] = current_ranking

            # 이전 랭킹과 비교하여 ranking_change 계산
            previous_rank = previous_ranking.get(item["song_info_id"], None)
            if previous_rank is None:
                item["ranking_change"] = 0  # 새로운 노래는 변동 없음
                item["new"] = 1
            else:
                item["ranking_change"] = previous_rank - current_ranking
                item["new"] = 0

        return current_data
    
    def v2_init(self):
        try: 
            # 현재 시각과 이후 시각의 데이터를 확인
            seoul_tz = ZoneInfo('Asia/Seoul')
            now = datetime.now(seoul_tz)
            one_hour_later = now + timedelta(hours=1)
            formatted_string_for_one_hour_later = one_hour_later.strftime("%Y-%m-%d-%H-Hot_Trend")

            nextInitKey = f"{formatted_string_for_one_hour_later}_MIXED_ALL"
            current = now.strftime("%Y-%m-%d-%H-Hot_Trend")
            currentInitKey = f"{current}_MIXED_ALL"

            rdb = self.setup_redis_config()
            currentExists = rdb.exists(currentInitKey)
            nextExists = rdb.exists(nextInitKey)
            rdb.close()

            if not currentExists:
                # 다음 시각의 데이터가 없는 경우 -> 일단 넣어둠
                db = self.setup_db_config()
                cursor = db.cursor()
                cursor.execute("""     
                    WITH scored_songs AS (
                        SELECT
                            ma.song_info_id,
                            SUM(ma.action_score) AS total_score,
                            ma.gender,
                            CASE
                                WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 10 AND 19 THEN '10'
                                WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 20 AND 29 THEN '20'
                                WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 30 AND 39 THEN '30'
                                WHEN YEAR(CURDATE()) - ma.birthyear + 1 > 39 THEN '40+'
                            END AS age_group
                        FROM member_action as ma
                        WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 1 MONTH)
                        GROUP BY ma.song_info_id, ma.gender, age_group
                    )
                    SELECT
                        ss.song_info_id,
                        ss.total_score,
                        s.song_name,
                        s.artist_name,
                        s.song_number,
                        s.is_mr,
                        s.album,
                        ss.gender,
                        ss.age_group
                    FROM scored_songs ss
                    JOIN song_info s ON ss.song_info_id = s.song_info_id
                """)
                results = cursor.fetchall()
                db.close()

                # is_live 추가 - 임시
                for _, item in enumerate(results):
                    item["is_live"] = 0

                rdb = self.setup_redis_config()

                # Redis에서 이전 데이터 가져오기
                seoul_tz = ZoneInfo('Asia/Seoul')
                now = datetime.now(seoul_tz)
                one_hour_before = now - timedelta(hours=1)
                formatted_string_for_one_hour_before = one_hour_before.strftime("%Y-%m-%d-%H-Hot_Trend")
                time_key = f"{formatted_string_for_one_hour_before}_MIXED_ALL"

                previous_data = {"male": {}, "female": {}, "mixed": {}}
                for gender_key in ["MALE", "FEMALE", "MIXED"]:
                    for age_group in ["ALL", "10", "20", "30", "40+"]:
                        key = f"{time_key}_{gender_key}_{age_group}"
                        if rdb.exists(key):
                            current_data = rdb.get(key)
                            previous_data[gender_key.lower()][age_group] = json.loads(current_data)
                        else:
                            previous_data[gender_key.lower()][age_group] = []

                # 성별 및 나이대별로 데이터를 분류하기 위한 저장소
                male_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
                female_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
                mixed_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}

                # 데이터를 성별 및 나이대에 맞게 분류
                for row in results:
                    gender = row['gender']
                    age_group = row['age_group']
                    mixed_data['ALL'].append(row)
                    mixed_data[age_group].append(row)
                    if gender == 'MALE':
                        male_data['ALL'].append(row)
                        male_data[age_group].append(row)
                    elif gender == 'FEMALE':
                        female_data['ALL'].append(row)
                        female_data[age_group].append(row)

                # 각 케이스에 대해 상위 20개의 노래만 자르고 ranking과 ranking_change 추가
                for age_group in male_data.keys():
                    male_data[age_group] = self.add_ranking_info(self.get_top_20_by_score(male_data[age_group]), previous_data["male"][age_group])
                    female_data[age_group] = self.add_ranking_info(self.get_top_20_by_score(female_data[age_group]), previous_data["female"][age_group])
                    mixed_data[age_group] = self.add_ranking_info(self.get_top_20_by_score(mixed_data[age_group]), previous_data["mixed"][age_group])

                # Redis에 저장할 키 생성
                seoul_tz = ZoneInfo('Asia/Seoul')
                now = datetime.now(seoul_tz)
                formatted_string_for_current_time = now.strftime("%Y-%m-%d-%H-Hot_Trend")

                # Redis에 저장 (남성, 여성, 성별미정)
                for age_group in male_data.keys():
                    male_key = f"{formatted_string_for_current_time}_MALE_{age_group}"
                    female_key = f"{formatted_string_for_current_time}_FEMALE_{age_group}"
                    combined_key = f"{formatted_string_for_current_time}_MIXED_{age_group}"

                    rdb.set(male_key, json.dumps(male_data[age_group]))
                    rdb.expire(male_key, 4210)

                    rdb.set(female_key, json.dumps(female_data[age_group]))
                    rdb.expire(female_key, 4210)

                    rdb.set(combined_key, json.dumps(mixed_data[age_group]))
                    rdb.expire(combined_key, 4210)

                rdb.close()
                logger.info("현재시각 hot trending 추가 완료")

            # 다음 시각의 데이터가 없는 경우 -> 일단 넣어둠
            if not nextExists:
                self.v2_scheduler()
                logger.info("다음시각 hot trending 추가 완료")

        except Exception as e:
            logger.exception("hot trending 갱신 중 오류 발생")