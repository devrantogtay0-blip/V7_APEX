# -*- coding: utf-8 -*-
"""
🐋 V7 APEX — NOTIFICATIONS
Telegram · Discord · E-posta · Webhook · Konsol (fallback)
Tüm kanallar opsiyoneldir; env yoksa konsola yazar
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import threading
import time
import urllib.request
from dataclasses import dataclass
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

from v7_config import CACHE_DIR

log = logging.getLogger("v7.notifications")

NOTIFY_LOG = os.path.join(CACHE_DIR, "v7_notifications.jsonl")


# ═══════════════════════════════════════════════════════════════════
# BİLDİRİM KAYDI
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Notification:
    """Gönderilen bildirim."""
    channel: str
    subject: str
    body: str
    status: str  # sent / failed / skipped
    ts: str = ""

    def __post_init__(self):
        if not self.ts:
            self.ts = time.strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel, "subject": self.subject,
            "body": self.body[:200], "status": self.status, "ts": self.ts,
        }


# ═══════════════════════════════════════════════════════════════════
# TELEGRAM
# ═══════════════════════════════════════════════════════════════════

class TelegramChannel:
    """Telegram Bot API ile bildirim."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)

    def send(self, subject: str, body: str) -> bool:
        """Mesaj gönder."""
        if not self.enabled:
            return False
        try:
            text = f"*{subject}*\n\n{body}"
            url = f"https://api.telegram.org/bot{self.token}/sendMessage"
            data = json.dumps({
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "Markdown",
            }).encode("utf-8")
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status == 200
        except Exception as e:
            log.warning("Telegram gönderim hatası: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════
# DISCORD
# ═══════════════════════════════════════════════════════════════════

class DiscordChannel:
    """Discord Webhook ile bildirim."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")
        self.enabled = bool(self.url)

    def send(self, subject: str, body: str) -> bool:
        """Webhook'a mesaj gönder."""
        if not self.enabled:
            return False
        try:
            data = json.dumps({
                "content": f"**{subject}**\n{body}",
            }).encode("utf-8")
            req = urllib.request.Request(
                self.url, data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return 200 <= r.status < 300
        except Exception as e:
            log.warning("Discord gönderim hatası: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════
# E-POSTA
# ═══════════════════════════════════════════════════════════════════

class EmailChannel:
    """SMTP ile e-posta."""

    def __init__(self) -> None:
        self.host = os.environ.get("SMTP_HOST", "")
        self.port = int(os.environ.get("SMTP_PORT", "587"))
        self.user = os.environ.get("SMTP_USER", "")
        self.passwd = os.environ.get("SMTP_PASS", "")
        self.to = os.environ.get("SMTP_TO", "")
        self.enabled = bool(self.host and self.user and self.passwd and self.to)

    def send(self, subject: str, body: str) -> bool:
        """E-posta gönder."""
        if not self.enabled:
            return False
        try:
            msg = MIMEText(body, "plain", "utf-8")
            msg["Subject"] = subject
            msg["From"] = self.user
            msg["To"] = self.to

            with smtplib.SMTP(self.host, self.port, timeout=10) as s:
                s.starttls()
                s.login(self.user, self.passwd)
                s.send_message(msg)
            return True
        except Exception as e:
            log.warning("E-posta gönderim hatası: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════
# WEBHOOK (Generic)
# ═══════════════════════════════════════════════════════════════════

class WebhookChannel:
    """Generic webhook (POST JSON)."""

    def __init__(self, url: Optional[str] = None):
        self.url = url or os.environ.get("WEBHOOK_URL", "")
        self.enabled = bool(self.url)

    def send(self, subject: str, body: str) -> bool:
        """JSON POST."""
        if not self.enabled:
            return False
        try:
            data = json.dumps({
                "subject": subject, "body": body,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }).encode("utf-8")
            req = urllib.request.Request(
                self.url, data=data,
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                return 200 <= r.status < 300
        except Exception as e:
            log.warning("Webhook hatası: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════
# KONSOL (fallback)
# ═══════════════════════════════════════════════════════════════════

class ConsoleChannel:
    """Konsola ve dosyaya yazar."""

    def __init__(self, file_path: str = NOTIFY_LOG):
        self.file_path = file_path
        self.enabled = True
        try:
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        except Exception:
            pass

    def send(self, subject: str, body: str) -> bool:
        """Konsol + dosya."""
        try:
            line = f"[{time.strftime('%H:%M:%S')}] {subject}: {body}"
            print(line)
            with open(self.file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "subject": subject, "body": body,
                }, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            log.warning("Konsol yazma hatası: %s", e)
            return False


# ═══════════════════════════════════════════════════════════════════
# NOTIFICATION MANAGER
# ═══════════════════════════════════════════════════════════════════

class NotificationManager:
    """Tüm bildirim kanallarını yönetir."""

    def __init__(self) -> None:
        self.channels: Dict[str, Any] = {
            "telegram": TelegramChannel(),
            "discord": DiscordChannel(),
            "email": EmailChannel(),
            "webhook": WebhookChannel(),
            "console": ConsoleChannel(),
        }
        self.history: List[Notification] = []
        self._lock = threading.RLock()

    def send(self, subject: str, body: str,
             channels: Optional[List[str]] = None) -> List[Notification]:
        """
        Bildirim gönder.
        channels: None → tüm aktif kanallar
        """
        targets = channels or list(self.channels.keys())
        out: List[Notification] = []
        for name in targets:
            ch = self.channels.get(name)
            if ch is None:
                continue
            if not getattr(ch, "enabled", False):
                n = Notification(name, subject, body, "skipped")
            else:
                ok = ch.send(subject, body)
                n = Notification(name, subject, body, "sent" if ok else "failed")
            with self._lock:
                self.history.append(n)
                if len(self.history) > 1000:
                    self.history = self.history[-500:]
            out.append(n)
        return out

    def active_channels(self) -> List[str]:
        """Aktif kanallar."""
        return [k for k, v in self.channels.items() if getattr(v, "enabled", False)]

    def recent(self, n: int = 20) -> List[Dict]:
        """Son n bildirim."""
        with self._lock:
            return [x.to_dict() for x in self.history[-n:]]

    def stats(self) -> Dict[str, int]:
        """Kanal başına gönderim."""
        with self._lock:
            stats: Dict[str, int] = {}
            for n in self.history:
                stats[n.status] = stats.get(n.status, 0) + 1
        return stats


# ═══════════════════════════════════════════════════════════════════
# KISAYOLLAR
# ═══════════════════════════════════════════════════════════════════

_manager: Optional[NotificationManager] = None
_lock = threading.RLock()


def get_notifier() -> NotificationManager:
    """Singleton."""
    global _manager
    with _lock:
        if _manager is None:
            _manager = NotificationManager()
        return _manager


def notify(subject: str, body: str,
           channels: Optional[List[str]] = None) -> List[Notification]:
    """Kısayol."""
    return get_notifier().send(subject, body, channels)


# ═══════════════════════════════════════════════════════════════════
# KENDİNİ TEST
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🐋 V7 APEX — Notifications Test")

    # 1) Singleton
    nm = get_notifier()
    print(f"\n✅ Aktif kanallar: {nm.active_channels()}")
    assert "console" in nm.active_channels()

    # 2) Console bildirim (offline)
    res = nm.send("Test", "Bu bir test bildirimidir.", channels=["console"])
    print(f"\n✅ Konsol bildirim: {len(res)}, status={res[0].status}")
    assert res[0].status == "sent"

    # 3) Kapalı kanal
    res2 = nm.send("Test2", "Telegram testi", channels=["telegram"])
    print(f"✅ Telegram (kapalı): status={res2[0].status}")
    assert res2[0].status == "skipped"

    # 4) Tüm kanallar
    res3 = nm.send("Tümü", "Tüm kanallara test")
    print(f"✅ Tüm kanallar: {len(res3)}")
    for r in res3:
        print(f"   - {r.channel}: {r.status}")

    # 5) Stats
    st = nm.stats()
    print(f"\n✅ İstatistik: {st}")

    # 6) Son bildirimler
    recent = nm.recent(5)
    print(f"\n✅ Son {len(recent)} bildirim")
    for r in recent[-3:]:
        print(f"   [{r['channel']}] {r['subject']} ({r['status']})")

    # 7) Kısayol
    res4 = notify("Kısayol", "notify() fonksiyonu testi", channels=["console"])
    print(f"\n✅ Kısayol çalışıyor: {len(res4)}")

    print("\n✅ Notifications testi geçti")
