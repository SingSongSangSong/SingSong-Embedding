# 베이스 이미지로 Python 3.9을 사용
FROM python:3.9-slim

# 작업 디렉토리를 생성하고 설정
WORKDIR /app

# 필요에 따라 시스템 패키지 설치 (예: gcc, libpq-dev 등)
# RUN apt-get update && apt-get install -y gcc

# 로컬 파일을 컨테이너의 작업 디렉토리에 복사
COPY . .

# 의존성 설치 (예: requirements.txt가 있다면)
RUN pip install --no-cache-dir -r requirements.txt

# gRPC 서버 실행 (main.py가 있는 경우)
CMD ["python", "main.py"]