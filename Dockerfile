# Python 베이스 이미지
FROM python:3.9-slim

# 작업 디렉터리 설정
WORKDIR /app

# 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# gRPC 서버 포트 노출
EXPOSE 50051

# gRPC 서버 실행
CMD ["sh", "-c", "ddtrace-run python main.py"]