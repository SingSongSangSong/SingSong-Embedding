# SingSong-Embedding

## 실행 순서
1. milvus를 docker run해야함
    standalone_embed.sh start
2. milvus insight를 보고 싶다면 추가적이니 docker run
3. python3 milvus_data_384.insert.py
4. python main.py
5. SingSong-GRPC를 run하면 매 55분에 파이썬 서버에 호출하여 db업데이트를 한다!

