import logging

import os
import pymysql
from datetime import datetime
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TJCrawlingService:
    def __init__(self):
        self.db_host = os.getenv('DB_HOST')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.db_database = os.getenv('DB_DATABASE')
        self.db_port = 3306

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

        logger.info("db 연결 성공")
        return db
     
    def save_to_db(self, songs):
        connection = self.setup_db_config()
        cursor = connection.cursor()

        # song_info 테이블에 데이터 삽입
        insert_query = """
            INSERT IGNORE INTO song_info (song_number, song_name, artist_name, is_mr, is_live) VALUES (%s, %s, %s, %s, %s)
        """
        inserted_rows = 0

        for song in songs:
            cursor.execute(insert_query, song)
            if cursor.rowcount > 0:  # 데이터가 삽입된 경우에만 카운팅
                inserted_rows += cursor.rowcount

        connection.commit()
        cursor.close()
        connection.close()

        logger.info(f"{inserted_rows}개의 신곡 정보가 성공적으로 데이터베이스에 저장 되었습니다.")

     
    def crawl_new_songs(self):
        now = datetime.now()
        year = now.strftime("%Y")  # 현재 연도 (YYYY)
        month = now.strftime("%m")  # 현재 달 (MM)
        logger.info(f"{year}년 {month}월의 신곡을 크롤링 시작합니다.")
        url = f"https://m.tjmedia.com/tjsong/song_monthNew.asp?YY={year}&MM={month}"

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # 크롤링할 데이터가 들어 있는 태그를 찾아서 반복문으로 처리
        songs = []
        for row in soup.select('tr')[1:]:  # 첫 번째 tr은 헤더이므로 제외
            cols = row.find_all('td')
            if len(cols) >= 3:
                song_number = cols[0].text.strip()
                song_name = cols[1].text.strip()
                artist_name = cols[2].text.strip()
                songs.append((song_number, song_name, artist_name))
        
        logger.info(f"{len(songs)}개의 신곡 정보가 성공적으로 크롤링되었습니다.")
        logger.info(songs)
        return songs
    
    def crawl_and_save_new_songs(self):
        songs = self.crawl_new_songs()
        songs_include_mr_and_live = self.crawl_mr_and_live(songs)
        self.save_to_db(songs_include_mr_and_live)

    def crawl_mr_and_live(self, songs):
        return_songs = []
        for song in songs:
            return_songs.append(self.crawl_one_mr_and_live(song))
        logger.info(f"{len(songs)}개의 MR 및 Live 정보가 성공적으로 크롤링되었습니다.")
        return return_songs
    
    def crawl_one_mr_and_live(self, song):
        song_number = song[0]
        url = 'https://www.tjmedia.com/tjsong/song_search_list.asp?strType=16&natType=&strText='+str(song_number)+'&strCond=1&strSize05=100'

        # POST 요청 보내기
        response = requests.get(url)
        html = response.content.decode('utf-8', 'replace')
        
        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(html, 'html.parser')
        
        # 해당 위치의 song_number와 일치 여부 확인
        song_number_element = soup.select_one("#BoardType1 > table > tbody > tr:nth-child(2) > td:nth-child(1)")

        # song_number가 존재하고, 해당 요소의 텍스트와 일치하는지 확인
        if song_number_element and song_number_element.text.strip() == str(song_number):
            logger.info(f"곡 번호 {song_number}와 일치하는 곡이 발견되었습니다.")
        else:
            logger.info(f"곡 번호 {song_number}와 일치하는 곡을 찾을 수 없습니다.")
            return

        # 곡 정보가 담긴 테이블을 찾고, 태그 확인
        song_info = soup.find('table', {'class': 'board_type1'})

        # "live"와 "mr" 태그가 있는지 확인
        live_tag = song_info.find_all('img', {'src': '/images/tjsong/live_icon.png'})
        mr_tag = song_info.find_all('img', {'src': '/images/tjsong/mr_icon.png'})
        
        is_live = len(live_tag) > 0
        is_mr = len(mr_tag) > 0

        # print(f"곡 번호 {song_number} 검색 결과:")
        # print(f" - Live 태그: {'있음' if is_live else '없음'}")
        # print(f" - MR 태그: {'있음' if is_mr else '없음'}")

        return (song_number, song[1], song[2], is_mr, is_live)
