"""Player #XXXXX 스텁 선수들을 MLB Stats API로 실명 업데이트."""
import json
import sys
from urllib.request import urlopen

sys.path.insert(0, ".")

from sqlalchemy import text
from src.db.session import get_session
from src.common.logger import get_logger

logger = get_logger("fix_player_stubs")


def fix_stubs() -> None:
    with get_session() as session:
        rows = session.execute(
            text("SELECT mlbam_id FROM players WHERE full_name LIKE 'Player #%' OR full_name = 'TBD'")
        ).fetchall()

    player_ids = [r[0] for r in rows]
    if not player_ids:
        logger.info("업데이트할 스텁 없음")
        return

    logger.info("스텁 %d명 조회 중...", len(player_ids))

    # MLB Stats API 벌크 조회 (100명씩)
    id_to_name: dict[int, str] = {}
    batch_size = 100
    for i in range(0, len(player_ids), batch_size):
        batch = player_ids[i:i + batch_size]
        ids_str = ",".join(str(pid) for pid in batch)
        url = f"https://statsapi.mlb.com/api/v1/people?personIds={ids_str}"
        try:
            with urlopen(url, timeout=10) as res:
                data = json.loads(res.read().decode("utf-8"))
            for p in data.get("people", []):
                pid = p.get("id")
                name = p.get("fullName")
                if pid and name:
                    id_to_name[pid] = name
        except Exception as e:
            logger.warning("배치 %d 조회 실패: %s", i // batch_size, e)

    if not id_to_name:
        logger.warning("MLB API에서 이름을 가져오지 못함")
        return

    with get_session() as session:
        updated = 0
        for pid, name in id_to_name.items():
            result = session.execute(
                text("UPDATE players SET full_name=:n WHERE mlbam_id=:id AND (full_name LIKE 'Player #%' OR full_name='TBD')"),
                {"n": name, "id": pid},
            )
            updated += result.rowcount
        session.commit()

    logger.info("✅ %d명 실명 업데이트 완료 (API에서 %d명 조회)", updated, len(id_to_name))
    not_found = set(player_ids) - set(id_to_name.keys())
    if not_found:
        logger.info("이름 미확인 ID (%d명): %s", len(not_found), sorted(not_found))


if __name__ == "__main__":
    fix_stubs()
