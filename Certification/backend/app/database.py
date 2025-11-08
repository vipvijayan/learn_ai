"""
SQLite database setup and models for the School Assistant
"""
import sqlite3
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database path
DB_PATH = Path(__file__).parent.parent / "data" / "school_events.db"


def get_db_connection():
    """Create and return a database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # This enables column access by name
    return conn


def init_database():
    """Initialize the database with required tables"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                selected_school_id INTEGER,
                FOREIGN KEY (selected_school_id) REFERENCES schools (id)
            )
        """)
        
        # Create schools table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                location TEXT,
                email_suffix TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_schools junction table for many-to-many relationship
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_schools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                school_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                FOREIGN KEY (school_id) REFERENCES schools (id) ON DELETE CASCADE,
                UNIQUE(user_id, school_id)
            )
        """)
        
        # Create user_preferences table to store bookmarks and other preferences
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                preference_key TEXT NOT NULL,
                preference_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(user_id, preference_key)
            )
        """)
        
        # Create bookmarks table to store individual chat bookmarks
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bookmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                bookmark_id TEXT UNIQUE NOT NULL,
                message_type TEXT NOT NULL,
                message_content TEXT NOT NULL,
                message_context TEXT,
                message_source TEXT,
                message_index INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
        """)
        
        # Create index on email for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
        """)
        
        # Create indexes for user_schools junction table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_schools_user ON user_schools(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_schools_school ON user_schools(school_id)
        """)
        
        # Create indexes for user_preferences table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_preferences_user ON user_preferences(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_preferences_key ON user_preferences(user_id, preference_key)
        """)
        
        # Create indexes for bookmarks table
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bookmarks_id ON bookmarks(bookmark_id)
        """)
        
        conn.commit()
        logger.info(f"Database initialized successfully at {DB_PATH}")
        
        # Insert some sample schools if table is empty
        cursor.execute("SELECT COUNT(*) FROM schools")
        if cursor.fetchone()[0] == 0:
            sample_schools = [
                ("Round Rock ISD", "Round Rock, TX", "roundrockisd.org"),
                ("Austin ISD", "Austin, TX", "austinisd.org"),
                ("Pflugerville ISD", "Pflugerville, TX", "pfisd.net"),
                ("Leander ISD", "Leander, TX", "leanderisd.org"),
                ("Georgetown ISD", "Georgetown, TX", "georgetownisd.org"),
                ("Cedar Park", "Cedar Park, TX", "cpschools.com"),
                ("Hutto ISD", "Hutto, TX", "hutto.txed.net"),
                ("Manor ISD", "Manor, TX", "manorisd.net")
            ]
            cursor.executemany(
                "INSERT INTO schools (name, location, email_suffix) VALUES (?, ?, ?)",
                sample_schools
            )
            conn.commit()
            logger.info(f"Inserted {len(sample_schools)} sample schools")
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_or_create_user(email: str):
    """Get existing user or create new one"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Try to get existing user
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        
        if user:
            # Update last login
            cursor.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email = ?",
                (email,)
            )
            conn.commit()
            logger.info(f"User logged in: {email}")
        else:
            # Create new user
            cursor.execute(
                "INSERT INTO users (email, last_login) VALUES (?, CURRENT_TIMESTAMP)",
                (email,)
            )
            conn.commit()
            cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
            user = cursor.fetchone()
            logger.info(f"New user created: {email}")
        
        return dict(user)
        
    except Exception as e:
        logger.error(f"Error in get_or_create_user: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_all_schools():
    """Get all schools from the database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT * FROM schools ORDER BY name")
        schools = cursor.fetchall()
        return [dict(school) for school in schools]
    except Exception as e:
        logger.error(f"Error getting schools: {e}")
        raise
    finally:
        conn.close()


def update_user_school(email: str, school_id: int):
    """Update the selected school for a user (legacy single school support)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "UPDATE users SET selected_school_id = ? WHERE email = ?",
            (school_id, email)
        )
        conn.commit()
        logger.info(f"Updated school for user {email} to school_id {school_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating user school: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def add_user_school(email: str, school_id: int):
    """Add a school to user's selected schools"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user id
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user:
            raise ValueError(f"User not found: {email}")
        
        user_id = user['id']
        
        # Insert into user_schools (will be ignored if already exists due to UNIQUE constraint)
        cursor.execute("""
            INSERT OR IGNORE INTO user_schools (user_id, school_id) VALUES (?, ?)
        """, (user_id, school_id))
        conn.commit()
        logger.info(f"Added school {school_id} for user {email}")
        return True
    except Exception as e:
        logger.error(f"Error adding user school: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def remove_user_school(email: str, school_id: int):
    """Remove a school from user's selected schools"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user id
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user:
            raise ValueError(f"User not found: {email}")
        
        user_id = user['id']
        
        cursor.execute("""
            DELETE FROM user_schools WHERE user_id = ? AND school_id = ?
        """, (user_id, school_id))
        conn.commit()
        logger.info(f"Removed school {school_id} for user {email}")
        return True
    except Exception as e:
        logger.error(f"Error removing user school: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def set_user_schools(email: str, school_ids: list):
    """Set user's schools (replaces all existing selections)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user id
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user:
            raise ValueError(f"User not found: {email}")
        
        user_id = user['id']
        
        # Delete existing selections
        cursor.execute("DELETE FROM user_schools WHERE user_id = ?", (user_id,))
        
        # Insert new selections
        if school_ids:
            cursor.executemany(
                "INSERT INTO user_schools (user_id, school_id) VALUES (?, ?)",
                [(user_id, school_id) for school_id in school_ids]
            )
        
        # Also update the legacy selected_school_id to first school if available
        if school_ids:
            cursor.execute(
                "UPDATE users SET selected_school_id = ? WHERE email = ?",
                (school_ids[0], email)
            )
        else:
            cursor.execute(
                "UPDATE users SET selected_school_id = NULL WHERE email = ?",
                (email,)
            )
        
        conn.commit()
        logger.info(f"Set schools for user {email}: {school_ids}")
        return True
    except Exception as e:
        logger.error(f"Error setting user schools: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_user_schools(email: str):
    """Get all schools selected by a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT s.*
            FROM schools s
            INNER JOIN user_schools us ON s.id = us.school_id
            INNER JOIN users u ON us.user_id = u.id
            WHERE u.email = ?
            ORDER BY s.name
        """, (email,))
        schools = cursor.fetchall()
        return [dict(school) for school in schools]
    except Exception as e:
        logger.error(f"Error getting user schools: {e}")
        raise
    finally:
        conn.close()


def get_user_with_school(email: str):
    """Get user with their selected school information (legacy single school)"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT u.*, s.name as school_name, s.location as school_location, s.email_suffix as school_email_suffix
            FROM users u
            LEFT JOIN schools s ON u.selected_school_id = s.id
            WHERE u.email = ?
        """, (email,))
        user = cursor.fetchone()
        return dict(user) if user else None
    except Exception as e:
        logger.error(f"Error getting user with school: {e}")
        raise
    finally:
        conn.close()


def get_user_with_schools(email: str):
    """Get user with all their selected schools"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user info
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user:
            return None
        
        user_dict = dict(user)
        
        # Get all selected schools
        schools = get_user_schools(email)
        user_dict['schools'] = schools
        user_dict['school_count'] = len(schools)
        
        # Keep legacy fields for backwards compatibility
        if schools:
            user_dict['school_name'] = schools[0]['name']
            user_dict['school_location'] = schools[0]['location']
            user_dict['school_email_suffix'] = schools[0]['email_suffix']
        
        return user_dict
    except Exception as e:
        logger.error(f"Error getting user with schools: {e}")
        raise
    finally:
        conn.close()


# ============================================================
# USER PREFERENCES FUNCTIONS
# ============================================================

def set_user_preference(email: str, preference_key: str, preference_value: str):
    """Set or update a user preference"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user:
            raise Exception(f"User not found: {email}")
        
        user_id = user['id']
        
        # Insert or update preference
        cursor.execute("""
            INSERT INTO user_preferences (user_id, preference_key, preference_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, preference_key) 
            DO UPDATE SET preference_value = ?, updated_at = CURRENT_TIMESTAMP
        """, (user_id, preference_key, preference_value, preference_value))
        
        conn.commit()
        logger.info(f"Set preference for {email}: {preference_key}")
        return True
    except Exception as e:
        logger.error(f"Error setting user preference: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_user_preference(email: str, preference_key: str):
    """Get a specific user preference"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT preference_value 
            FROM user_preferences 
            JOIN users ON user_preferences.user_id = users.id
            WHERE users.email = ? AND user_preferences.preference_key = ?
        """, (email, preference_key))
        
        result = cursor.fetchone()
        return result['preference_value'] if result else None
    except Exception as e:
        logger.error(f"Error getting user preference: {e}")
        return None
    finally:
        conn.close()


def get_all_user_preferences(email: str):
    """Get all preferences for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT preference_key, preference_value, updated_at
            FROM user_preferences 
            JOIN users ON user_preferences.user_id = users.id
            WHERE users.email = ?
        """, (email,))
        
        preferences = {}
        for row in cursor.fetchall():
            preferences[row['preference_key']] = {
                'value': row['preference_value'],
                'updated_at': row['updated_at']
            }
        
        return preferences
    except Exception as e:
        logger.error(f"Error getting user preferences: {e}")
        return {}
    finally:
        conn.close()


# ============================================================
# BOOKMARKS FUNCTIONS
# ============================================================

def add_bookmark(email: str, bookmark_id: str, message_type: str, message_content: str, 
                 message_context: str = None, message_source: str = None, message_index: int = None):
    """Add a bookmark for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get user ID
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        if not user:
            raise Exception(f"User not found: {email}")
        
        user_id = user['id']
        
        # Insert bookmark
        cursor.execute("""
            INSERT INTO bookmarks (user_id, bookmark_id, message_type, message_content, 
                                   message_context, message_source, message_index)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, bookmark_id, message_type, message_content, 
              message_context, message_source, message_index))
        
        conn.commit()
        logger.info(f"Added bookmark for {email}: {bookmark_id}")
        return True
    except Exception as e:
        logger.error(f"Error adding bookmark: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def remove_bookmark(email: str, bookmark_id: str):
    """Remove a bookmark for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM bookmarks 
            WHERE bookmark_id = ? 
            AND user_id = (SELECT id FROM users WHERE email = ?)
        """, (bookmark_id, email))
        
        conn.commit()
        logger.info(f"Removed bookmark for {email}: {bookmark_id}")
        return True
    except Exception as e:
        logger.error(f"Error removing bookmark: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_user_bookmarks(email: str):
    """Get all bookmarks for a user"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT bookmarks.bookmark_id, bookmarks.message_type, bookmarks.message_content, 
                   bookmarks.message_context, bookmarks.message_source, bookmarks.message_index, 
                   bookmarks.created_at
            FROM bookmarks 
            JOIN users ON bookmarks.user_id = users.id
            WHERE users.email = ?
            ORDER BY bookmarks.created_at DESC
        """, (email,))
        
        bookmarks = []
        for row in cursor.fetchall():
            bookmarks.append({
                'bookmark_id': row['bookmark_id'],
                'message_type': row['message_type'],
                'message_content': row['message_content'],
                'message_context': row['message_context'],
                'message_source': row['message_source'],
                'message_index': row['message_index'],
                'created_at': row['created_at']
            })
        
        return bookmarks
    except Exception as e:
        logger.error(f"Error getting user bookmarks: {e}")
        return []
    finally:
        conn.close()


def reset_database():
    """
    Reset the database by dropping all tables and recreating them.
    WARNING: This will delete all data!
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        logger.warning("🚨 RESETTING DATABASE - ALL DATA WILL BE DELETED!")
        
        # Drop all tables (in correct order due to foreign key constraints)
        cursor.execute("DROP TABLE IF EXISTS bookmarks")
        cursor.execute("DROP TABLE IF EXISTS user_preferences")
        cursor.execute("DROP TABLE IF EXISTS user_schools")
        cursor.execute("DROP TABLE IF EXISTS users")
        cursor.execute("DROP TABLE IF EXISTS schools")
        
        # Drop indexes (they'll be recreated with tables)
        cursor.execute("DROP INDEX IF EXISTS idx_users_email")
        cursor.execute("DROP INDEX IF EXISTS idx_user_schools_user")
        cursor.execute("DROP INDEX IF EXISTS idx_user_schools_school")
        cursor.execute("DROP INDEX IF EXISTS idx_user_preferences_user")
        cursor.execute("DROP INDEX IF EXISTS idx_user_preferences_key")
        cursor.execute("DROP INDEX IF EXISTS idx_bookmarks_user")
        cursor.execute("DROP INDEX IF EXISTS idx_bookmarks_id")
        
        conn.commit()
        logger.info("✅ All tables dropped successfully")
        
        # Reinitialize database with fresh tables
        conn.close()
        init_database()
        
        logger.info("✅ Database reset complete - fresh tables created")
        return True
    except Exception as e:
        logger.error(f"❌ Error resetting database: {e}")
        conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


# Initialize database when module is imported
if __name__ == "__main__":
    init_database()
