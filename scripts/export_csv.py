# export_csv.py
import pandas as pd
from sqlalchemy import create_engine
import os

DB_URL = "postgresql+psycopg2://mlb_user:mlbpred123@localhost:5432/mlb_prediction"
engine = create_engine(DB_URL)

print("📦 DB에서 Colab용 CSV 추출을 시작합니다...")

# 연도별로 나눠서 추출
seasons = [2023,2024,2025,2026]

for year in seasons:
    print(f"⏳ {year}년 데이터 추출 중...")
    
    # 🌟 [수정 완료] DB의 'spin_rate'를 꺼내서 코랩이 아는 'release_spin_rate'로 이름표 변경!
    query = f"""
    SELECT 
        player_name, batter_id, home_team, away_team, inning_topbot, 
        release_speed, spin_rate AS release_spin_rate, launch_speed, launch_angle, description
    FROM statcast_pitches 
    WHERE EXTRACT(YEAR FROM game_date) = {year};
    """
    
    df = pd.read_sql(query, engine)
    
    if not df.empty:
        file_name = f"statcast_{year}.csv"
        df.to_csv(file_name, index=False)
        print(f"✅ {file_name} 저장 완료! ({len(df)}행)")
    else:
        print(f"⚠️ {year}년 데이터가 아직 없습니다.")

print("🎉 모든 추출이 끝났습니다! 생성된 CSV 파일들을 Colab에 업로드하세요.")