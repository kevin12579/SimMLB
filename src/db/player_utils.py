"""선수 FK 위반 방지를 위한 DB 유틸리티"""
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.db.models.players import Player
from src.common.logger import get_logger

logger = get_logger(__name__)


def ensure_players_exist(player_ids: set[int], session: Session) -> None:
    """DB에 없는 선수 ID를 최소 stub으로 일괄 삽입 (FK 위반 방지)"""
    if not player_ids:
        return
    existing = {
        row[0] for row in session.query(Player.mlbam_id)
        .filter(Player.mlbam_id.in_(player_ids))
        .all()
    }
    missing = player_ids - existing
    if not missing:
        return
    stubs = [
        dict(mlbam_id=pid, full_name=f"Player #{pid}", position="UNK", status="unknown")
        for pid in missing
    ]
    session.execute(
        insert(Player).values(stubs).on_conflict_do_nothing(index_elements=["mlbam_id"])
    )
    logger.debug("Inserted %d player stubs for unknown IDs", len(stubs))
