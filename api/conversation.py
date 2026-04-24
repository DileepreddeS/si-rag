from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from api.schemas import ChatMessage


class ConversationMemory:

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns  = max_turns
        self.memory_path = Path("data/memory.json")
        self.memory_path.parent.mkdir(parents=True, exist_ok=True)
        self._sessions: Dict[str, List[ChatMessage]] = self._load()

    def _load(self) -> Dict[str, List[ChatMessage]]:
        if self.memory_path.exists():
            try:
                raw = json.loads(self.memory_path.read_text())
                return {
                    sid: [ChatMessage(**m) for m in msgs]
                    for sid, msgs in raw.items()
                }
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        data = {
            sid: [m.dict() for m in msgs]
            for sid, msgs in self._sessions.items()
        }
        self.memory_path.write_text(json.dumps(data, indent=2))

    def get_history(self, session_id: str) -> List[ChatMessage]:
        return self._sessions.get(session_id, [])

    def add_turn(
        self,
        session_id: str,
        question: str,
        answer: str,
    ) -> None:
        if session_id not in self._sessions:
            self._sessions[session_id] = []

        self._sessions[session_id].append(
            ChatMessage(role="user", content=question)
        )
        self._sessions[session_id].append(
            ChatMessage(role="assistant", content=answer)
        )

        # Keep only last max_turns
        if len(self._sessions[session_id]) > self.max_turns * 2:
            self._sessions[session_id] = self._sessions[session_id][-(self.max_turns * 2):]

        self._save()

    def clear_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save()

    def format_history_for_prompt(self, session_id: str) -> str:
        history = self.get_history(session_id)
        if not history:
            return ""

        lines = ["Previous conversation:"]
        for msg in history:
            prefix = "User" if msg.role == "user" else "Assistant"
            lines.append(f"{prefix}: {msg.content}")

        return "\n".join(lines) + "\n\n"