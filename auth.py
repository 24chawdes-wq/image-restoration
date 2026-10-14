"""
auth.py - User Authentication & Session Management Module
Handles SQLite user accounts, hashed credentials, and restoration history.
"""

import os
import sqlite3
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes tables for users and their restoration history."""
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS restoration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filter_type TEXT NOT NULL,
                filename TEXT NOT NULL,
                sharpness_gain REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        """)
        conn.commit()


def create_user(name: str, email: str, password: str) -> dict:
    name = name.strip()
    email = email.strip().lower()
    if not name:
        return {"success": False, "error": "Name is required."}
    if not email or "@" not in email:
        return {"success": False, "error": "Please provide a valid email address."}
    if not password or len(password) < 6:
        return {"success": False, "error": "Password must be at least 6 characters long."}

    password_hash = generate_password_hash(password)

    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, password_hash)
            )
            conn.commit()
            user_id = cursor.lastrowid
            return {
                "success": True,
                "user": {
                    "id": user_id,
                    "name": name,
                    "email": email
                }
            }
    except sqlite3.IntegrityError:
        return {"success": False, "error": "An account with this email already exists. Please log in."}
    except Exception as e:
        return {"success": False, "error": f"Registration failed: {str(e)}"}


def verify_user(email: str, password: str) -> dict:
    email = email.strip().lower()
    if not email or not password:
        return {"success": False, "error": "Email and password are required."}

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, password_hash FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()

        if not row:
            return {"success": False, "error": "Invalid email or password."}

        if not check_password_hash(row["password_hash"], password):
            return {"success": False, "error": "Invalid email or password."}

        return {
            "success": True,
            "user": {
                "id": row["id"],
                "name": row["name"],
                "email": row["email"]
            }
        }


def get_user_by_id(user_id: int) -> dict | None:
    if not user_id:
        return None
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, email, created_at FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "name": row["name"],
                "email": row["email"],
                "created_at": str(row["created_at"])
            }
        return None


def record_restoration(user_id: int, filter_type: str, filename: str, sharpness_gain: float):
    if not user_id:
        return
    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO restoration_history (user_id, filter_type, filename, sharpness_gain) VALUES (?, ?, ?, ?)",
                (user_id, filter_type, filename, sharpness_gain)
            )
            conn.commit()
    except Exception:
        pass


def get_user_history(user_id: int, limit: int = 5) -> list[dict]:
    if not user_id:
        return []
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filter_type, filename, sharpness_gain, created_at FROM restoration_history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        return [
            {
                "filter_type": r["filter_type"],
                "filename": r["filename"],
                "sharpness_gain": r["sharpness_gain"],
                "created_at": str(r["created_at"])
            }
            for r in rows
        ]


# Initialize tables when imported
init_db()
