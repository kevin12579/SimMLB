"""팀 로스터와 선수 마스터를 MLB StatsAPI에서 동기화"""
from datetime import date

from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

from src.collector.mlb_statsapi_client import MLBStatsAPIClient
from src.db.models.players import Player
from src.db.models.teams import Team
from src.common.logger import get_logger

logger = get_logger(__name__)


async def sync_all_rosters(season: int, session: Session) -> None:
    """모든 팀 로스터를 순회하며 선수 마스터 동기화"""
    async with MLBStatsAPIClient() as client:
        teams = session.query(Team).all()
        for team in teams:
            try:
                roster = await client.fetch_roster(team.mlbam_team_id, season)
                for entry in roster:
                    p = entry.get("person", {})
                    pos = entry.get("position", {})
                    if not p.get("id"):
                        continue
                    stmt = insert(Player).values(
                        mlbam_id=p["id"],
                        full_name=p.get("fullName", "")[:100],
                        position=pos.get("abbreviation", "")[:5],
                        status="active",
                        current_team_id=team.mlbam_team_id,
                    ).on_conflict_do_update(
                        index_elements=["mlbam_id"],
                        set_=dict(
                            current_team_id=team.mlbam_team_id,
                            status="active",
                            position=pos.get("abbreviation", "")[:5],
                        ),
                    )
                    session.execute(stmt)
                logger.info("Synced roster for team %d (%d players)", team.mlbam_team_id, len(roster))
            except Exception as e:
                logger.error("Failed roster sync for team %d: %s", team.mlbam_team_id, e)
        session.commit()
