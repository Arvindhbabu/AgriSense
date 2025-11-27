# utils/memory_bank.py
"""
Simple in-memory memory bank. For production, replace with persistent DB (Cloudant, Firestore, etc.)
"""
import threading

class MemoryBank:
    def __init__(self):
        self._lock = threading.Lock()
        self.users = {}

    def save_user(self, user_id: str, data: dict):
        with self._lock:
            self.users[user_id] = data.copy()

    def get_user(self, user_id: str):
        with self._lock:
            return self.users.get(user_id)

    def delete_user(self, user_id: str):
        with self._lock:
            if user_id in self.users:
                del self.users[user_id]
