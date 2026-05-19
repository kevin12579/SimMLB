"""백필 후 데이터 무결성 검증 스크립트"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, ".")

from src.db.session import get_session
from src.db.models.teams import Team
from src.db.models.players import Player
from src.db.models.games import Game, StatcastPitch
from src.db.models.players import PlayerSeasonStats
from src.common.logger import get_logger

logger = get_logger("verify")


def verify() -> None:
    with get_session() as session:
        team_count = session.query(Team).count()
        player_count = session.query(Player).count()
        game_count = session.query(Game).count()
        finished_count = session.query(Game).filter(Game.status == "Final").count()
        stat_count = session.query(PlayerSeasonStats).count()
        statcast_count = session.query(StatcastPitch).count()

        print(f"{'=' * 40}")
        print(f"  팀:             {team_count:>8,}")
        print(f"  선수:           {player_count:>8,}")
        print(f"  총 경기:        {game_count:>8,}  (기대: ~4,860)")
        print(f"  완료 경기:      {finished_count:>8,}")
        print(f"  선수 시즌 통계: {stat_count:>8,}")
        print(f"  Statcast 투구:  {statcast_count:>8,}")
        print(f"{'=' * 40}")

        # 간단 경고
        if game_count < 4000:
            print("  [WARN]  경기 수가 예상보다 적습니다. 백필을 확인하세요.")
        else:
            print("  [OK] 경기 수 정상")

        if team_count < 30:
            print("  [WARN]  팀 수가 30개 미만입니다.")
        else:
            print("  [OK] 팀 수 정상")


if __name__ == "__main__":
    verify()
