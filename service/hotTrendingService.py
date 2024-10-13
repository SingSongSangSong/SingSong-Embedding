import json
import logging
import os
import aiomysql
import aioredis
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

    async def setup_db_config(self):
        try:
            # 비동기 MySQL 연결 설정
            pool = await aiomysql.create_pool(
                host=self.db_host,
                user=self.db_user,
                password=self.db_password,
                db=self.db_database,
                port=self.db_port,
                charset='utf8mb4',
                cursorclass=aiomysql.DictCursor,
                autocommit=True  # 자동 커밋 설정
            )
            logger.info("DB 연결 성공")
            return pool

        except aiomysql.MySQLError as e:
            logger.error(f"MySQL 연결 실패: {e}")
            raise
    
    async def setup_redis_config(self):
        try:
            # 비동기 Redis 클라이언트 생성
            redis = await aioredis.from_url(
                f"redis://{self.redis_host}:{self.redis_port}"
            )
            
            # Redis 연결 확인
            await redis.ping()
            logger.info("Redis 연결 성공")
            return redis

        except (aioredis.ConnectionError, aioredis.RedisError) as e:
            logger.error(f"Redis 연결 실패: {e}")
            raise

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
                        WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 2 WEEK)
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
                        WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 2 WEEK)
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
    async def v2_scheduler(self):
        try: 
            pool = await self.setup_db_config()
            async with pool.acquire() as conn:
                async with conn.cursor() as cursor:
                    await cursor.execute("""     
                        WITH scored_songs AS (
                            SELECT
                                ma.song_info_id,
                                SUM(ma.action_score) AS total_score,
                                ma.gender,
                                CASE
                                    WHEN ma.birthyear = 0 THEN 'ALL'
                                    WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 10 AND 19 THEN '10'
                                    WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 20 AND 29 THEN '20'
                                    WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 30 AND 39 THEN '30'
                                    WHEN YEAR(CURDATE()) - ma.birthyear + 1 > 39 THEN '40+'
                                    ELSE 'ALL'
                                END AS age_group
                            FROM member_action as ma
                            WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 2 WEEK)
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
                            ss.age_group,
                            s.melon_song_id
                        FROM scored_songs ss
                        JOIN song_info s ON ss.song_info_id = s.song_info_id
                    """)
                    results = await cursor.fetchall()

            redis = await self.setup_redis_config()
            # Redis에서 이전 데이터 가져오기
            seoul_tz = ZoneInfo('Asia/Seoul')
            now = datetime.now(seoul_tz)
            formatted_string_for_current_time = now.strftime("%Y-%m-%d-%H-Hot_Trend")

            for result in results:
                result["is_live"] = 0  # 임시 처리

            previous_data = {"male": {}, "female": {}, "mixed": {}}
            for gender_key in ["MALE", "FEMALE", "MIXED"]:
                for age_group in ["ALL", "10", "20", "30", "40+"]:
                    key = f"{formatted_string_for_current_time}_{gender_key}_{age_group}"
                    data = await redis.get(key)
                    if data:
                        try:
                            previous_data[gender_key.lower()][age_group] = json.loads(data)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON for key: {key}. Setting as empty list.")
                            previous_data[gender_key.lower()][age_group] = []
                    else:
                        previous_data[gender_key.lower()][age_group] = []

            # 성별 및 나이대별로 데이터를 분류하기 위한 저장소
            male_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
            female_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
            mixed_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}

            # 데이터 분류
            for row in results:
                gender, age_group = row["gender"], row["age_group"]

                # 전체 성별 - 전체 연령
                self.update_data(mixed_data["ALL"], row, 'MIXED', 'ALL')

                # 전체 성별 - 특정 연령
                if age_group != 'ALL':
                    self.update_data(mixed_data[age_group], row, 'MIXED')

                # 특정 성별 - 전체 연령
                if gender == 'MALE':
                    self.update_data(male_data["ALL"], row)
                elif gender == 'FEMALE':
                    self.update_data(female_data["ALL"], row)

                # 특정 성별 - 특정 연령
                if gender == 'MALE' and age_group != 'ALL':
                    male_data[age_group].append(row.copy())
                elif gender == 'FEMALE' and age_group != 'ALL':
                    female_data[age_group].append(row.copy())

            # 상위 20개의 노래와 랭킹 정보 추가
            for age_group in male_data.keys():
                male_data[age_group] = self.add_ranking_info(
                    self.get_top_20_by_score(male_data[age_group]), 
                    previous_data["male"][age_group]
                )
                female_data[age_group] = self.add_ranking_info(
                    self.get_top_20_by_score(female_data[age_group]), 
                    previous_data["female"][age_group]
                )
                mixed_data[age_group] = self.add_ranking_info(
                    self.get_top_20_by_score(mixed_data[age_group]), 
                    previous_data["mixed"][age_group]
                )

            # 한 시간 후에 만료될 Redis 키 설정
            one_hour_later = now + timedelta(hours=1)
            formatted_string_for_one_hour_later = one_hour_later.strftime("%Y-%m-%d-%H-Hot_Trend")

            # Redis에 데이터 저장
            await self.save_to_redis(redis, formatted_string_for_one_hour_later, male_data, "MALE")
            await self.save_to_redis(redis, formatted_string_for_one_hour_later, female_data, "FEMALE")
            await self.save_to_redis(redis, formatted_string_for_one_hour_later, mixed_data, "MIXED")

            redis.close()
            logger.info("hot trending 갱신 성공")

        except KeyError as e:
            logger.error(f"hot trending 갱신 중 KeyError 발생: {str(e)}")
        except Exception as e:
            logger.exception("hot trending 갱신 중 오류 발생")

    def update_data(self, data_list, row, gender='MIXED', age_group=None):
        existing_song = next((item for item in data_list if item["song_info_id"] == row["song_info_id"]), None)
        if existing_song:
            existing_song["total_score"] += row["total_score"]
        else:
            new_row = row.copy()
            new_row["gender"] = gender
            if age_group:
                new_row["age_group"] = age_group
            data_list.append(new_row)

    def get_top_20_by_score(self, data_list):
        return sorted(data_list, key=lambda x: x["total_score"], reverse=True)[:20]

    def add_ranking_info(self, current_data, previous_data):
        # 이전 데이터에서 'ranking'이 없으면 기본값 float('inf')로 설정
        previous_ranking = {item["song_info_id"]: item.get("ranking", float('inf')) for item in previous_data}

        for idx, item in enumerate(current_data):
            item["ranking"] = idx + 1  # 현재 데이터의 순위 설정
            previous_rank = previous_ranking.get(item["song_info_id"], float('inf'))

            # 이전 랭킹이 없으면 'new'로 표시하고 ranking_change는 0으로 설정
            if previous_rank == float('inf'):
                item["ranking_change"] = 0
                item["new"] = 1  # 새로 추가된 노래
            else:
                item["ranking_change"] = previous_rank - item["ranking"]
                item["new"] = 0  # 기존에 존재하던 노래

        return current_data

    async def save_to_redis(self, redis, base_key, data, gender):
        for age_group, items in data.items():
            key = f"{base_key}_{gender}_{age_group}"
            await redis.set(key, json.dumps(items))
            await redis.expire(key, 4800)
    
    async def v2_init(self):
        try:
            redis = await self.setup_redis_config()

            # 현재 및 다음 시간의 Redis 키 생성
            seoul_tz = ZoneInfo('Asia/Seoul')
            now = datetime.now(seoul_tz)
            one_hour_later = now + timedelta(hours=1)

            formatted_string_for_current_time = now.strftime("%Y-%m-%d-%H-Hot_Trend")
            formatted_string_for_one_hour_later = one_hour_later.strftime("%Y-%m-%d-%H-Hot_Trend")

            currentInitKey = f"{formatted_string_for_current_time}_MIXED_ALL"
            nextInitKey = f"{formatted_string_for_one_hour_later}_MIXED_ALL"

            currentExists = await redis.exists(currentInitKey)
            nextExists = await redis.exists(nextInitKey)

            if not currentExists:
                pool = await self.setup_db_config()
                async with pool.acquire() as conn:
                    async with conn.cursor() as cursor:
                        await cursor.execute("""     
                            WITH scored_songs AS (
                                SELECT
                                    ma.song_info_id,
                                    SUM(ma.action_score) AS total_score,
                                    ma.gender,
                                    CASE
                                        WHEN ma.birthyear = 0 THEN 'ALL'
                                        WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 10 AND 19 THEN '10'
                                        WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 20 AND 29 THEN '20'
                                        WHEN YEAR(CURDATE()) - ma.birthyear + 1 BETWEEN 30 AND 39 THEN '30'
                                        WHEN YEAR(CURDATE()) - ma.birthyear + 1 > 39 THEN '40+'
                                        ELSE 'ALL'
                                    END AS age_group
                                FROM member_action as ma
                                WHERE ma.CREATED_AT > DATE_SUB(NOW(), INTERVAL 2 WEEK)
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
                                ss.age_group,
                                s.melon_song_id
                            FROM scored_songs ss
                            JOIN song_info s ON ss.song_info_id = s.song_info_id
                        """)
                        results = await cursor.fetchall()

                for item in results:
                    item["is_live"] = 0

                previous_data = {"male": {}, "female": {}, "mixed": {}}
                one_hour_before = now - timedelta(hours=1)
                formatted_string_for_one_hour_before = one_hour_before.strftime("%Y-%m-%d-%H-Hot_Trend")

                for gender_key in ["MALE", "FEMALE", "MIXED"]:
                    for age_group in ["ALL", "10", "20", "30", "40+"]:
                        key = f"{formatted_string_for_one_hour_before}_{gender_key}_{age_group}"
                        data = await redis.get(key)
                        if data:
                            try:
                                previous_data[gender_key.lower()][age_group] = json.loads(data)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON for key: {key}. Setting as empty list.")
                                previous_data[gender_key.lower()][age_group] = []
                        else:
                            previous_data[gender_key.lower()][age_group] = []

                male_data, female_data, mixed_data = self.classify_data(results)

                await self.save_to_redis_for_init(redis, formatted_string_for_current_time, male_data, female_data, mixed_data)
                logger.info("현재 시각의 hot trending 추가 완료")

            if not nextExists:
                await self.v2_scheduler()
                logger.info("다음 시각의 hot trending 추가 완료")

            await redis.close()

        except KeyError as e:
            logger.error(f"hot trending 갱신 중 KeyError 발생: {str(e)}")
        except Exception as e:
            logger.exception("hot trending 갱신 중 오류 발생")

    def classify_data(self, results):
        male_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
        female_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}
        mixed_data = {"ALL": [], "10": [], "20": [], "30": [], "40+": []}

        for row in results:
            gender = row['gender']
            age_group = row['age_group']

            self.update_data(mixed_data["ALL"], row, 'MIXED', 'ALL')
            if age_group != 'ALL':
                self.update_data(mixed_data[age_group], row, 'MIXED')

            if gender == 'MALE':
                self.update_data(male_data["ALL"], row)
                if age_group != 'ALL':
                    male_data[age_group].append(row.copy())
            elif gender == 'FEMALE':
                self.update_data(female_data["ALL"], row)
                if age_group != 'ALL':
                    female_data[age_group].append(row.copy())

        return male_data, female_data, mixed_data

    def update_data(self, data_list, row, gender='MIXED', age_group=None):
        existing_song = next((item for item in data_list if item["song_info_id"] == row["song_info_id"]), None)
        if existing_song:
            existing_song["total_score"] += row["total_score"]
        else:
            new_row = row.copy()
            new_row["gender"] = gender
            if age_group:
                new_row["age_group"] = age_group
            data_list.append(new_row)

    async def save_to_redis_for_init(self, redis, base_key, male_data, female_data, mixed_data):
        for age_group in male_data.keys():
            male_key = f"{base_key}_MALE_{age_group}"
            female_key = f"{base_key}_FEMALE_{age_group}"
            mixed_key = f"{base_key}_MIXED_{age_group}"

            await redis.set(male_key, json.dumps(male_data[age_group]))
            await redis.expire(male_key, 4210)

            await redis.set(female_key, json.dumps(female_data[age_group]))
            await redis.expire(female_key, 4210)

            await redis.set(mixed_key, json.dumps(mixed_data[age_group]))
            await redis.expire(mixed_key, 4210)