import os
from sqlalchemy import create_engine
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

# .env 파일 로드
load_dotenv()

DATABASE_URL = f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_DATABASE')}"

# SQLAlchemy 엔진 및 세션 설정
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class DatabaseConfig:
    def __init__(self, host, user, password, database, charset='utf8mb4'):
        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.charset = charset

    def connect(self):
        try:
            engine = create_engine(f"mysql+pymysql://{self.user}:{self.password}@{self.host}/{self.database}")
            return engine
        except Exception as e:
            print(f"Error connecting to the database: {e}")
            raise

# DB 연결을 위한 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()