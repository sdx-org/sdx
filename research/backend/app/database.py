"""Database configuration and session management for research application."""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / 'data'
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / 'db.sqlite'
SQLALCHEMY_DATABASE_URL = f'sqlite:///{DB_PATH}'

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
