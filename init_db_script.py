from app import app
from database import init_db

with app.app_context():
    init_db()

print("Database initialized successfully!")
