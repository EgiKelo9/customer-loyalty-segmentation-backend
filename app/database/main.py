import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import get_settings

setting = get_settings()
DATABASE = f"{setting.DATABASE_URL}{setting.DATABASE_NAME}"
engine = create_engine(DATABASE)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Get a session for interacting with the database"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
def create_db():
    setting = get_settings()
    
    try:
        default_db_url = setting.DATABASE_URL + "postgres"
        db_engine = create_engine(default_db_url, isolation_level="AUTOCOMMIT")
        
        init_sql_path = os.path.join(os.path.dirname(__file__), "init.sql")
        with open(init_sql_path, "r", encoding="utf-8") as file:
            sql_script = file.read()
        
        with db_engine.connect() as connection:
            statements = [stmt.strip() for stmt in sql_script.split(';') if stmt.strip()]
            for statement in statements:
                try:
                    connection.execute(text(statement))
                except Exception as stmt_error:
                    print(f"Warning: {stmt_error}")
        
        print("Database and tables created successfully.")
    except Exception as e:
        print(f"Error executing init.sql: {e}")
