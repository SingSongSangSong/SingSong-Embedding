import json
import logging
import os

import pymysql
import redis
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # Python 3.9 이상
from dotenv import load_dotenv
from apscheduler.schedulers.background import BackgroundScheduler 

# .env 파일 로드
load_dotenv()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class HotTrendingSchedulingService:
    def __init__(self):
        self.db_host = os.getenv('RDSEndpoint')
        self.db_user = os.getenv('RDSUsername')
        self.db_password = os.getenv('RDSPassword')
        self.db_database = os.getenv('RDSName')
        self.db_port = 3306
        
        self.redis_host = os.getenv('ElastiCacheEndpoint')
        self.redis_port = 6379
        self.db, self.rdb = self.setup_config()

    def setup_config(self):
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

        try:
            rdb = redis.Redis(
                host=self.redis_host,
                port=self.redis_port
            )
            rdb.ping()
            logger.info("Redis 연결 성공")
            print("redis 연결 됨.")
        except redis.ConnectionError as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise

        logger.info("MySQL 및 Redis 연결 성공")

        print(rdb.info())
        return db, rdb

    # V1: 남성 전체, 여성 전체
    def v1_scheduler(self):
        try:
            cursor = self.db.cursor()

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

            seoul_tz = ZoneInfo('Asia/Seoul')
            now = datetime.now(seoul_tz)

            one_hour_later = now + timedelta(hours=1)
            formatted_string_for_one_hour_later = one_hour_later.strftime("%Y-%m-%d-%H-Hot_Trend")

            try:
                formatted_string_for_current_time = now.strftime("%Y-%m-%d-%H-Hot_Trend")

                male_exists = self.rdb.exists(formatted_string_for_current_time + "_MALE")
                female_exists = self.rdb.exists(formatted_string_for_current_time + "_FEMALE")

                if male_exists:
                    current_male_data = self.rdb.get(formatted_string_for_current_time + "_MALE")
                    male_json_data = json.loads(current_male_data)
                else:
                    male_json_data = []

                if female_exists:
                    current_female_data = self.rdb.get(formatted_string_for_current_time + "_FEMALE")
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
                print("실패")
                logger.error(f"현재 데이터를 Redis에서 가져오는 데 실패했습니다: {e}")
                raise

            new_formatted_string_for_male = formatted_string_for_one_hour_later + "_MALE"
            self.rdb.set(new_formatted_string_for_male, json.dumps(male_results))
            self.rdb.expire(new_formatted_string_for_male, 3910)

            new_formatted_string_for_female = formatted_string_for_one_hour_later + "_FEMALE"
            self.rdb.set(new_formatted_string_for_female, json.dumps(female_results))
            self.rdb.expire(new_formatted_string_for_female, 3910)

        except Exception as e:
            print("실패")
            logger.error(f"실행 중 오류 발생: {e}")
        
    # V2
    # 성별미정 전체, 10대, 20대, 30대, 40대 이상
    # 남성 전체, 10대, 20대, 30대, 40대 이상
    # 여성 전체, 10대, 20대, 30대, 40대 이상
    def v2_scheduler(self):
        print("성공!")