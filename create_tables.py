from src.db.base import Base 
from src.db.session import engine 
from src.db.models.players import Player, PlayerSeasonStats 
from src.db.models.teams import Team 
from src.db.models.games import Game 
from src.db.models.predictions import GamePrediction 
Base.metadata.create_all(bind=engine) 
print('완료') 
