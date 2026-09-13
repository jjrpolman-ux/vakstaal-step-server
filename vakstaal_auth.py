"""Single-admin authentication. Opaque, revocable sessions; no browser-stored secrets.

Database callbacks use the application's existing PostgreSQL connection. SQLite
is supported only for isolated tests. No CAD imports are needed to test security.
"""
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from contextlib import closing

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

COOKIE = "__Host-vakstaal_session"
SESSION_SECONDS = 8 * 60 * 60
ACCESS_SECONDS = 5 * 60
ITERATIONS = 600_000
LOG = logging.getLogger("vakstaal.auth")
TOKEN = re.compile(r"^[A-Za-z0-9_-]{43}$")


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_password_hash(value):
    return bool(re.fullmatch(r"pbkdf2_sha256\$600000\$[0-9a-f]{32}\$[0-9a-f]{64}", value))


def verify_password(password, encoded):
    if not valid_password_hash(encoded) or not isinstance(password, str):
        return False
    _, count, salt, expected = encoded.split("$")
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(count))
    return hmac.compare_digest(actual.hex(), expected)


def reply(status, detail, **extra):
    return JSONResponse({"ok": status < 400, "detail": detail, **extra}, status_code=status,
                        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})


class AuthStore:
    def __init__(self, connect, postgres):
        self.connect = connect
        self.postgres = postgres
        with closing(connect()) as conn:
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS admin_sessions (id TEXT PRIMARY KEY, expires BIGINT NOT NULL, credential TEXT NOT NULL)")
            cur.execute("CREATE TABLE IF NOT EXISTS admin_access_tokens (id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES admin_sessions(id) ON DELETE CASCADE, expires BIGINT NOT NULL)")
            cur.execute("CREATE TABLE IF NOT EXISTS admin_login_limits (bucket BIGINT PRIMARY KEY, attempts INTEGER NOT NULL)")
            cur.execute("CREATE INDEX IF NOT EXISTS admin_access_session ON admin_access_tokens(session_id)")
            conn.commit()

    def sql(self, value):
        return value.replace("?", "%s") if self.postgres() else value

    def attempt_allowed(self):
        # A global one-admin limit cannot be bypassed by rotating IP headers,
        # usernames or workers. It recovers automatically after five minutes.
        bucket = int(time.time()) // 300
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            cur.execute(self.sql("DELETE FROM admin_login_limits WHERE bucket < ?"), (bucket - 1,))
            cur.execute(self.sql("INSERT INTO admin_login_limits (bucket,attempts) VALUES (?,1) ON CONFLICT(bucket) DO UPDATE SET attempts=admin_login_limits.attempts+1 RETURNING attempts"), (bucket,))
            count = cur.fetchone()[0]
            conn.commit()
            return count <= 8

    def new_session(self, credential):
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            cur.execute(self.sql("DELETE FROM admin_access_tokens WHERE expires<=? OR session_id IN (SELECT id FROM admin_sessions WHERE expires<=?)"), (now, now))
            cur.execute(self.sql("DELETE FROM admin_sessions WHERE expires<=?"), (now,))
            cur.execute(self.sql("INSERT INTO admin_sessions(id,expires,credential) VALUES (?,?,?)"),
                        (fingerprint(token), now + SESSION_SECONDS, fingerprint(credential)))
            conn.commit()
        return token

    def session(self, raw, credential, bearer=False):
        if not raw or not TOKEN.fullmatch(raw):
            return None
        now = int(time.time())
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            if bearer:
                cur.execute(self.sql("SELECT s.id,s.expires FROM admin_sessions s JOIN admin_access_tokens a ON a.session_id=s.id WHERE a.id=? AND a.expires>? AND s.expires>? AND s.credential=?"),
                            (fingerprint(raw), now, now, fingerprint(credential)))
            else:
                cur.execute(self.sql("SELECT id,expires FROM admin_sessions WHERE id=? AND expires>? AND credential=?"),
                            (fingerprint(raw), now, fingerprint(credential)))
            row = cur.fetchone()
            return (row[0], row[1]) if row else None

    def access(self, session):
        raw = secrets.token_urlsafe(32)
        now = int(time.time())
        expires = min(now + ACCESS_SECONDS, session[1])
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            cur.execute(self.sql("DELETE FROM admin_access_tokens WHERE expires<=?"), (now,))
            cur.execute(self.sql("INSERT INTO admin_access_tokens(id,session_id,expires) VALUES (?,?,?)"),
                        (fingerprint(raw), session[0], expires))
            conn.commit()
        return raw, expires

    def revoke(self, raw):
        if not raw or not TOKEN.fullmatch(raw):
            return
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            cur.execute(self.sql("DELETE FROM admin_access_tokens WHERE session_id=?"), (fingerprint(raw),))
            cur.execute(self.sql("DELETE FROM admin_sessions WHERE id=?"), (fingerprint(raw),))
            conn.commit()


def install_auth(app, connect, postgres):
    store = AuthStore(connect, postgres)
    origins = {os.getenv("VAKSTAAL_APP_ORIGIN", "https://vakstaal-calculator.vercel.app").rstrip("/")}

    def credential():
        return os.getenv("VAKSTAAL_ADMIN_PASSWORD_HASH", "")

    def public(request):
        p, m = request.url.path, request.method
        return ((p == "/health" and m in {"GET", "HEAD"}) or
                (p == "/api/dropbox/oauth/callback" and m == "GET") or
                (re.fullmatch(r"/approve/[a-f0-9]{44}", p) and m == "GET") or
                (re.fullmatch(r"/approve/[a-f0-9]{44}/accept", p) and m == "POST"))

    @app.middleware("http")
    async def guard(request: Request, call_next):
        # OPTIONS carries no business data; CORSMiddleware validates its origin.
        if request.method == "OPTIONS":
            return await call_next(request)
        if public(request):
            response = await call_next(request)
        else:
            encoded = credential()
            if not valid_password_hash(encoded):
                return reply(503, "Beveiliging is nog niet ingesteld. Neem contact op met de beheerder.")
            if request.method not in {"GET", "HEAD"} and request.headers.get("origin") not in origins:
                return reply(403, "Dit verzoek komt niet van het vaste calculatoradres.")
            p = request.url.path
            if p in {"/api/auth/login", "/api/auth/session", "/api/auth/logout"}:
                response = await call_next(request)
            else:
                auth = request.headers.get("authorization", "")
                try:
                    session = await run_in_threadpool(store.session, auth[7:] if auth.startswith("Bearer ") else "", encoded, True)
                except Exception:
                    LOG.error("Session validation unavailable")
                    return reply(503, "Sessiecontrole tijdelijk niet beschikbaar. Probeer opnieuw.")
                if not session:
                    return reply(401, "Log opnieuw in om verder te gaan.")
                response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        return response

    @app.post("/api/auth/login")
    async def login(request: Request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                return reply(413, "Inlogverzoek is te groot.")
        try:
            data = json.loads(body)
        except (ValueError, UnicodeError):
            return reply(400, "Ongeldig inlogverzoek.")
        if not isinstance(data, dict) or not isinstance(data.get("password"), str):
            return reply(400, "Vul je wachtwoord in.")
        if not await run_in_threadpool(store.attempt_allowed):
            response = reply(429, "Te veel inlogpogingen. Wacht vijf minuten en probeer opnieuw.")
            response.headers["Retry-After"] = "300"
            return response
        encoded = credential()
        if not await run_in_threadpool(verify_password, data["password"], encoded):
            return reply(401, "Wachtwoord is niet juist.")
        await run_in_threadpool(store.revoke, request.cookies.get(COOKIE))
        raw = await run_in_threadpool(store.new_session, encoded)
        response = reply(200, "Ingelogd.")
        response.set_cookie(COOKIE, raw, max_age=SESSION_SECONDS, secure=True, httponly=True, samesite="strict", path="/")
        return response

    @app.get("/api/auth/session")
    def session(request: Request):
        current = store.session(request.cookies.get(COOKIE), credential())
        if not current:
            return reply(401, "Log opnieuw in om verder te gaan.")
        # The HttpOnly session stays on Vercel. Only a five-minute token goes
        # into JS memory for direct large uploads to Render (never localStorage).
        token, expires = store.access(current)
        return reply(200, "Ingelogd.", access_token=token, expires_at=expires,
                     session_expires_at=current[1])

    @app.post("/api/auth/logout")
    def logout(request: Request):
        store.revoke(request.cookies.get(COOKIE))
        response = reply(200, "Uitgelogd.")
        response.delete_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="strict")
        return response

    return store
