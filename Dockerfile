# Python 베이스 이미지
FROM python:3.9-slim

# 작업 디렉터리 설정
WORKDIR /app

# Install dependencies for awscli
RUN apt-get update && apt-get install -y \
    python3-pip \
    && pip3 install awscli

# 종속성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the dataframe directory
RUN mkdir -p /app/dataframe

# Accept AWS credentials as build arguments
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_DEFAULT_REGION
ARG S3_BUCKET_NAME

# Set AWS credentials as environment variables inside the build process
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION}
ENV S3_BUCKET_NAME=${S3_BUCKET_NAME}

# Download the CSV files from S3 during the build process
RUN aws s3 cp s3://${S3_BUCKET_NAME}/song_info.csv /app/dataframe/song_info.csv
RUN aws s3 cp s3://${S3_BUCKET_NAME}/with_ssss_22286_updated5.csv /app/dataframe/with_ssss_22286_updated5.csv

# 애플리케이션 코드 복사
COPY . .

# gRPC 서버 포트 노출
EXPOSE 50051

# gRPC 서버 실행
CMD ["python", "main.py"]