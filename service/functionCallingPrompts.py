from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List
import logging


class ExtractCommonTraits(BaseModel):
    genre: str
    year: str
    country: str
    artist_type: str
    artist_gender: str
    situation: List[str]
    octave: str
    vocal_range: str
    lyrics: str

class PromptsForFunctionCalling:
    def __init__(self, query: str):
        self.prompt_for_decision = [
                {"role": "system", "content": """
                You are an assistant that categorizes user inputs into specific types of song recommendations and extracts important information.

                Follow these guidelines for classifying the query into one of the following nine categories and extracting relevant information, ensuring all relevant fields are extracted even if the information is not provided.

                1. **Single Song/Artist Query:**
                    - If the user input contains either a specific song title, an artist name, or both, asking for similar songs.
                    - If both a song title and artist name are mentioned, treat it as a full song-artist query.
                    - If only a song title or artist name is mentioned, return recommendations based on the available information.
                    - **Extract**: song name, artist name (If not provided, return an empty list `[]` or None).
                    - **For example**: 
                        - "버즈의 가시나 같은 노래 추천해줘".
                        - "추억은 만남보다 이별에 남아 같은 노래 찾아줘".
                        - "엠씨더맥스 노래 추천해줘".
                    - Output query type as 'single_song_artist'.

                2. **Multiple Song-Artist Pairs Query:**
                    - If the user provides multiple song-artist pairs or only song titles or artist names and is asking for recommendations based on the common features between those pairs.
                    - **Extract**: all song names, all artist names (If not provided, return an empty list `[]` or None).
                    - **For example**:
                        - "Recommend me songs like 'Bang Bang Bang' by BigBang and 'Thorn' by Buzz".
                        - "마이클 잭슨의 Thriller랑 퀸의 Bohemian Rhapsody 같은 곡 추천해줘".
                        - "아이유와 태연의 노래 추천해줘".
                        - "가시랑 거짓말 같은 노래 추천해줘".
                    - Output query type as 'multiple_song_artist'.

                3. **Octave/Key-based Query:**
                    - If the user mentions specific octaves 
                    - The query might involve asking for songs in a particular octave or maximum/minimum octave.
                    - **Extract**: octave information(as MAX 3옥타브레 or EQUAL 2옥타브시 or MIN 1옥타브레), genre (If not provided, return `None`), gender (If not provided, return `None`), year (If not provided, return `None`).
                    - **Octave Information**: Octave information should be extracted in the following format:
                        - MAX [pitch]: The song should have a maximum pitch of [pitch].
                        - MIN [pitch]: The song should have a minimum pitch of [pitch].
                    - **For example**:
                        - "최고음 2옥타브시 노래 추천해줘".
                        - "남자 1옥타브로 노래 추천해줘".
                    - Output query type as 'octave_key'.

                4. **Octave with Song/Artist Query:**
                    - If the user provides both a specific song and mentions octaves or key changes.
                    - **Extract**: song name, artist name, octave information (If not provided, return an empty list `[]` or `None`).
                    - **For example**:
                        - "김연우의 '내가 너의 곁에 잠시 살았다는걸' 정도의 음역대의 노래 추천해줘".
                        - "Lower the key for 'Imagine' by John Lennon".
                    - Output query type as 'song_artist_octave'.

                5. **Vocal Range (High/Low) Query:**
                    - If the user asks for songs based on vocal range, such as high-pitched or low-pitched songs or difficulty related to singing (e.g., easy or hard songs).
                    - **Extract**: vocal range (high/low) (If not provided, return `None`).
                    - **For example**:
                        - "고음 자신 있는데, 고음 폭발하는 곡 추천해줘".
                        - "남자 저음발라드 추천해줘".
                        - "음역대가 낮은 남자 노래 추천해줘".
                    - Output query type as 'vocal_range'.

                6. **Situation-Based Query (breakup, carol, ssum, rainy, military, wedding):**
                    - If the user is asking for songs based on specific situations or occasions.
                    - **Extract**: situation context (e.g., breakup, carol, ssum, etc.), gender if applicable (If not provided, return `None`).
                    - **For example**:
                        - "감성적인 이별 노래 추천해줘".
                        - "여자친구 앞에서 부를만한 노래".
                        - "오늘 헤어졌는데 부를만한 노래 추천좀..".
                        - "썸탈때 부를만한 남자노래 추천해줘".
                    - **Situation Keywords**: 반드시 다음 목록의 상황 키워드로 반환되어야 합니다:
                        - 그시절 띵곡 -> classics
                        - 썸 -> ssum
                        - 이별/헤어졌을때 -> breakup
                        - 크리스마스/캐롤/눈 -> carol
                        - 마지막곡 -> finale
                        - 신나는/춤/흥 -> dance
                        - 듀엣 -> duet
                        - 비/꿀꿀할때 -> rainy
                        - 회사생활 -> office
                        - 축하 -> wedding
                        - 군대/입대 -> military
                    - Output query type as 'situation'.

                7. **Year/Gender/Genre-based/Solo-Group Based Query:**
                    - If the user asks for songs based on a specific year, gender, genre, or whether it's suitable for solo or group singing.
                    - If gender exists in the query You have to decide which gender that the user wants to get recommendations for.
                    - **Extract**: year, genre, gender (female/male/mixed), performance type (solo/group) (If not provided, return `None` or an empty list).
                    - When extracting **genre**, return the most appropriate match from the following list of genres from the database:
                        - 국악
                        - 발라드
                        - 록/메탈
                        - 댄스
                        - 성인가요/트로트
                        - 포크/블루스
                        - 키즈
                        - 창작동요
                        - 국내영화
                        - 국내드라마
                        - 랩/힙합
                        - R&B/Soul
                        - 인디음악
                        - 애니메이션/웹툰
                        - 만화
                        - 교과서동요
                        - 국외영화
                        - POP
                        - 클래식
                        - 크로스오버
                        - J-POP
                        - CCM
                        - 게임
                        - 컨트리
                        - 재즈
                        - 보컬재즈
                        - 포크
                        - 블루스
                        - 일렉트로니카
                        - 월드뮤직
                        - 애시드/퓨전/팝
                        - 국내뮤지컬
                    - **For example**:
                        - "2010년도 쯤에 신나는 노래 추천해줘".
                        - "발라드 2024".
                        - "혼자 노래방 갈 때 부르기 좋은 노래 있을까?".
                        - "여자 가수들의 힙합곡 추천".
                    - Output query type as 'year_gender_genre'.

                It's important to ensure that the query fits into only one of these nine categories. If the user input is unclear, make your best effort to infer the most likely category.
                Always extract all the relevant fields, even if the information is missing, by returning `None` or an empty list where applicable.

                Format:
                - Query Type: <single_song_artist/multiple_song_artist/octave_key/song_artist_octave/hit_songs/vocal_range/situation/year_gender_genre>
                - Song Name: [<song_name1>, <song_name2>, ...] (If applicable, otherwise `[]`)
                - Artist Name: [<artist_name1>, <artist_name2>, ...] (If applicable, otherwise `[]`)
                - Octave: <octave_info>(`MAX [pitch]`/`EQUAL [pitch]`/`MIN [pitch]`) (If applicable, otherwise `None`)
                - Vocal Range: <vocal_range> (high, low, or `None`)
                - Gender: <gender_info> (female, male, mixed, or `None`)
                - Year: <year_info> (If year range, return `year >= start && year <= end`, If specific year, return `year == input_year` otherwise `None`)
                - Genre: <genre_info> (If applicable, otherwise `None`)
                - Situation: [<situation_info>] (classics, ssum, breakup, carol, finale, dance, duet, rainy, office, wedding, military) (If applicable, otherwise `None`)
                """},
                {"role": "user", "content": query}
            ]
        
class ExtractCommonTraitService:
    def __init__(self, asyncOpenai):
        self.asyncOpenai = asyncOpenai
    async def ExtractCommonTraitService(self, data):
        # Step 4: Create the detailed prompt for multiple song-artist pairs
        prompt_template = PromptTemplate.from_template(
            """
            you are a music recommendation assistant. The user is asking for several songs so you have to extract the common features and create the query based on that.

            {data}

            You have to extract the common features and gotta suggest what kinds of feature that users want to get. You have to consider the following aspects:

            - Genre(국악, 발라드, 록/메탈, 댄스, 성인가요/트로트, 포크/블루스, 키즈, 창작동요, 국내영화, 국내드라마, 랩/힙합, R&B/Soul, 인디음악, 애니메이션/웹툰, 만화, 교과서동요, 국외영화, POP, 클래식, 크로스오버, J-POP, CCM, 게임, 컨트리, 재즈, 보컬재즈, 포크, 블루스, 일렉트로니카, 월드뮤직, 애시드/퓨전/팝, 국내뮤지컬)
            - Year
            - Country
            - Artist Type
            - Artist Gender
            - Situation(classics, ssum, breakup, carol, finale, dance, duet, rainy, office, wedding, military)
            - Octave
            - Vocal Range
            - Artist Name
            - Lyrics

            Format:
            - Genre: <genre> (if applicable, otherwise `None`)
            - Year: <year> (if applicable, otherwise `None`)
            - Country: <country> (if applicable, otherwise `None`)
            - Artist Type: <artist_type> (if applicable, otherwise `None`)
            - Artist Gender: <artist_gender> (if applicable, otherwise `None`)
            - Situation: [<situation>] (classics, ssum, breakup, carol, finale, dance, duet, rainy, office, wedding, military) (If applicable, otherwise `[]`)
            - Octave: <octave>  (if applicable, otherwise `None`)
            - Vocal Range: <vocal_range> (if applicable, otherwise `None`)
            - Artist Name: <artist_name> (If applicable, otherwise `[]`)
            - Lyrics: <lyrics> (if applicable, otherwise `None`)
            """
        )
        try:
            # Step 5: Format the prompt with the combined query and retrieved data
            prompt = prompt_template.format(data=data)

            # Step 6: Use the LLM to refine the query
            response = await self.asyncOpenai.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            response_format=ExtractCommonTraits,
            )
            parsed_result = response.choices[0].message.parsed
            # Step 7: Return the refined query
            
            return parsed_result
        except Exception as e:
            logging.error(f"Error during execution: {str(e)}")
            return None