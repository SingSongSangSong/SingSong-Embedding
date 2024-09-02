import pymysql
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class DatabaseConfig:
    def __init__(self, host, user, password, database, charset='utf8mb4'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset

    def connect(self):
        try:
            connection = pymysql.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset
            )
            return connection
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            raise

# DB 연결을 위한 함수
def get_db():
    config = DatabaseConfig(
        host=os.getenv('DB_HOST'),
        user=os.getenv('DB_USER'),
        password=os.getenv('DB_PASSWORD'),
        database=os.getenv('DB_DATABASE')
    )
    connection = config.connect()
    try:
        yield connection
    finally:
        connection.close()