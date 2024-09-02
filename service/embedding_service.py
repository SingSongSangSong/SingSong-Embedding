from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Optional
from db.dbConfig import DatabaseConfig
import json

class EmbeddingService:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)  # 모델을 초기화

    def fetch_recent_active_members(self, db: Session):
        try:
            query = """
            SELECT DISTINCT member_id
            FROM member_action
            WHERE created_at >= NOW() - INTERVAL 1 HOUR;
            """
            return pd.read_sql(query, db)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching recent active members: {e}")

    def fetch_member_action_data(self, db: Session, member_ids: List[int]):
        try:
            query = f"""
            SELECT member_id, song_info_id, action_type, gender, birthyear, action_score, created_at
            FROM member_action
            WHERE member_id IN ({','.join(map(str, member_ids))});
            """
            return pd.read_sql(query, db)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching member action data: {e}")

    def load_total_data(self):
        try:
            return pd.read_csv("dataframe/song_audio_data_22000.csv", encoding="utf-8-sig")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading total data from CSV: {e}")

    def merge_data(self, member_action_data: pd.DataFrame, total_data: pd.DataFrame):
        try:
            member_action_data['song_info_id'] = member_action_data['song_info_id'].astype(float)
            return pd.merge(member_action_data, total_data, on='song_info_id', how='left')
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error merging data: {e}")

    def calculate_user_preferences(self, group: pd.DataFrame, weights: dict):
        try:
            genre_preference = defaultdict(float)
            year_preference = defaultdict(float)
            country_preference = defaultdict(float)
            singer_type_preference = defaultdict(float)
            ssss_preference = defaultdict(float)
            max_pitch_preference = defaultdict(float)

            for _, row in group.iterrows():
                action_weight = weights.get(row['action_type'], 1)
                if pd.notna(row['genre']):
                    genre_preference[row['genre']] += action_weight
                if pd.notna(row['release_year']):
                    decade = self.convert_year_to_decade(row['release_year'])
                    year_preference[decade] += action_weight
                if pd.notna(row['country']):
                    country_preference[row['country']] += action_weight
                if pd.notna(row['singer_type']):
                    singer_type_preference[row['singer_type']] += action_weight
                if pd.notna(row['ssss']):
                    ssss_preference[row['ssss']] += action_weight
                if pd.notna(row['max_pitch']):
                    note_octave = self.convert_max_pitch_to_note_octave(row['max_pitch'])
                    max_pitch_preference[note_octave] += action_weight

            return genre_preference, year_preference, country_preference, singer_type_preference, ssss_preference, max_pitch_preference
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error calculating user preferences: {e}")

    def create_user_preference_sentence(self, genre_pref, year_pref, country_pref, singer_type_pref, ssss_pref, max_pitch_pref):
        try:
            user_preferences_parts = []

            top_genres = self.get_top_n_preferences(genre_pref)
            if top_genres:
                user_preferences_parts.append(f"The user likes the {top_genres[0]} genre the most, but also enjoys {', '.join(top_genres[1:])} genres.")

            top_years = self.get_top_n_preferences(year_pref)
            if top_years:
                user_preferences_parts.append(f"Prefers songs from the {top_years[0]} primarily, with an interest in the {', '.join(top_years[1:])} as well.")

            top_countries = self.get_top_n_preferences(country_pref)
            if top_countries:
                user_preferences_parts.append(f"Enjoys music from {top_countries[0]}, and also likes songs from {', '.join(top_countries[1:])}.")

            top_singer_types = self.get_top_n_preferences(singer_type_pref)
            if top_singer_types:
                user_preferences_parts.append(f"Prefers {top_singer_types[0]} artists, but is also inclined towards {', '.join(top_singer_types[1:])}.")

            top_ssss = self.get_top_n_preferences(ssss_pref)
            if top_ssss:
                user_preferences_parts.append(f"Is inclined towards songs with a {top_ssss[0]} mood, and also enjoys {', '.join(top_ssss[1:])} moods.")

            top_max_pitches = self.get_top_n_preferences(max_pitch_pref)
            if top_max_pitches:
                user_preferences_parts.append(f"Prefers songs with a max pitch of {top_max_pitches[0]}, but also enjoys {', '.join(top_max_pitches[1:])} pitches.")

            return " ".join(user_preferences_parts)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error creating user preference sentence: {e}")

    def embed_user_preferences(self, sentence: str):
        try:
            return self.model.encode(sentence)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error embedding user preferences: {e}")

    def update_or_insert_user_profile(self, db: Session, member_id: int, embedding: np.ndarray):
        try:
            with db.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) FROM user_profile WHERE member_id = %s", (member_id,))
                profile_exists = cursor.fetchone()[0] > 0

                if profile_exists:
                    update_query = """
                    UPDATE user_profile
                    SET embedding = %s
                    WHERE member_id = %s
                    """
                    cursor.execute(update_query, (json.dumps(embedding.tolist()), member_id))
                else:
                    insert_query = """
                    INSERT INTO user_profile (member_id, embedding)
                    VALUES (%s, %s)
                    """
                    cursor.execute(insert_query, (member_id, json.dumps(embedding.tolist())))

                db.commit()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating or inserting user profile for member_id {member_id}: {e}")

    def hz_to_midi(self, hz: float):
        try:
            return 69 + 12 * np.log2(hz / 440.0)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting Hz to MIDI: {e}")

    def midi_to_note_octave(self, midi: float):
        try:
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            note = int(midi) % 12
            octave = int(midi // 12) - 1
            return note_names[note], octave
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting MIDI to note and octave: {e}")

    def convert_max_pitch_to_note_octave(self, max_pitch: float):
        try:
            if pd.notna(max_pitch):
                midi_note = self.hz_to_midi(max_pitch)
                note, octave = self.midi_to_note_octave(midi_note)
                return f"{note}{octave}"
            else:
                return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting max pitch to note and octave: {e}")

    def convert_year_to_decade(self, year: int):
        try:
            if pd.notna(year):
                decade = (year // 10) * 10
                return f"{decade}s"
            else:
                return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error converting year to decade: {e}")

    def get_top_n_preferences(self, preferences: dict, n=3):
        try:
            sorted_preferences = sorted(preferences.items(), key=lambda item: item[1], reverse=True)
            return [pref[0] for pref in sorted_preferences[:n]]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error getting top {n} preferences: {e}")