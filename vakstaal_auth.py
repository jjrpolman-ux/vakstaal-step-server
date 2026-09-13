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
import smtplib
import ssl
import time
from contextlib import closing
from email.message import EmailMessage
from urllib.parse import urlsplit

from fastapi import Request, BackgroundTasks
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

COOKIE = "__Host-vakstaal_session"
SESSION_SECONDS = 8 * 60 * 60
ACCESS_SECONDS = 5 * 60
ITERATIONS = 600_000
LOG = logging.getLogger("vakstaal.auth")
TOKEN = re.compile(r"^[A-Za-z0-9_-]{43}$")
RESET_SECONDS = 15 * 60


def admin_email():
    return os.getenv("VAKSTAAL_ADMIN_EMAIL", "info@vakstaal.nl").strip().lower()


def utf8(value):
    try:
        return value.encode('utf-8') if isinstance(value, str) else b''
    except UnicodeError:
        return b''


def hash_password(password):
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), ITERATIONS)
    return f"pbkdf2_sha256${ITERATIONS}${salt}${digest.hex()}"


def send_auth_mail(subject, text):
    # SSL certificate verification is mandatory; no plaintext fallback.
    message = EmailMessage()
    message["From"] = os.environ["VAKSTAAL_SMTP_FROM"]
    message["To"] = admin_email()
    message["Subject"] = subject
    message.set_content(text)
    with smtplib.SMTP_SSL(os.environ["VAKSTAAL_SMTP_HOST"],
                          int(os.getenv("VAKSTAAL_SMTP_PORT", "465")),
                          timeout=10, context=ssl.create_default_context()) as smtp:
        smtp.login(os.environ["VAKSTAAL_SMTP_USER"], os.environ["VAKSTAAL_SMTP_PASSWORD"])
        smtp.send_message(message)


def send_change_notice():
    try:
        send_auth_mail("Je Vakstaal-wachtwoord is gewijzigd",
                       "Je beheerderswachtwoord voor de Vakstaal calculator is gewijzigd.\n"
                       "Alle bestaande sessies zijn beëindigd.\n\n"
                       "Was jij dit niet? Neem direct contact op met je beheerder en beveilig je mailbox.\n")
    except Exception:
        LOG.error("Password change notification could not be sent")


def fingerprint(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def valid_password_hash(value):
    return bool(re.fullmatch(r"pbkdf2_sha256\$600000\$[0-9a-f]{32}\$[0-9a-f]{64}", value))


def verify_password(password, encoded):
    if not valid_password_hash(encoded) or not isinstance(password, str):
        return False
    _, count, salt, expected = encoded.split("$")
    password_bytes = utf8(password)
    if not password_bytes:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password_bytes, bytes.fromhex(salt), int(count))
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
            cur.execute("CREATE TABLE IF NOT EXISTS admin_credentials (id INTEGER PRIMARY KEY, bootstrap TEXT NOT NULL, password_hash TEXT NOT NULL)")
            cur.execute("INSERT INTO admin_credentials(id,bootstrap,password_hash) VALUES (1,'','') ON CONFLICT(id) DO NOTHING")
            cur.execute("CREATE TABLE IF NOT EXISTS admin_reset_tokens (id TEXT PRIMARY KEY, expires BIGINT NOT NULL, owner TEXT NOT NULL)")
            cur.execute("CREATE TABLE IF NOT EXISTS admin_auth_limits (purpose TEXT NOT NULL, bucket BIGINT NOT NULL, attempts INTEGER NOT NULL, PRIMARY KEY(purpose,bucket))")
            conn.commit()

    def sql(self, value):
        return value.replace("?", "%s") if self.postgres() else value

    @staticmethod
    def effective(row, bootstrap):
        if not valid_password_hash(bootstrap):
            return ""
        return row[1] if row and row[0] == fingerprint(bootstrap) else bootstrap

    def credential(self, bootstrap):
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            cur.execute("SELECT bootstrap,password_hash FROM admin_credentials WHERE id=1")
            return self.effective(cur.fetchone(), bootstrap)

    def lock_account(self, conn):
        cur = conn.cursor()
        if not self.postgres():
            cur.execute("BEGIN IMMEDIATE")
        cur.execute("SELECT bootstrap,password_hash FROM admin_credentials WHERE id=1" +
                    (" FOR UPDATE" if self.postgres() else ""))
        return cur, cur.fetchone()

    def limited(self, purpose, limit=8, seconds=300):
        bucket = int(time.time()) // seconds
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            cur.execute(self.sql("DELETE FROM admin_auth_limits WHERE purpose=? AND bucket<?"), (purpose, bucket - 1))
            cur.execute(self.sql("INSERT INTO admin_auth_limits(purpose,bucket,attempts) VALUES (?,?,1) ON CONFLICT(purpose,bucket) DO UPDATE SET attempts=admin_auth_limits.attempts+1 RETURNING attempts"), (purpose, bucket))
            count = cur.fetchone()[0]
            conn.commit()
            return count <= limit

    def new_reset(self, bootstrap, email):
        raw = secrets.token_urlsafe(32)
        with closing(self.connect()) as conn:
            cur, row = self.lock_account(conn)
            encoded = self.effective(row, bootstrap)
            if not valid_password_hash(encoded):
                return None
            now = int(time.time())
            cur.execute(self.sql("DELETE FROM admin_reset_tokens WHERE expires<=?"), (now,))
            cur.execute(self.sql("INSERT INTO admin_reset_tokens(id,expires,owner) VALUES (?,?,?)"),
                        (fingerprint(raw), now + RESET_SECONDS, fingerprint(encoded + '\n' + email)))
            conn.commit()
        return raw

    def discard_reset(self, raw):
        with closing(self.connect()) as conn:
            cur = conn.cursor()
            cur.execute(self.sql("DELETE FROM admin_reset_tokens WHERE id=?"), (fingerprint(raw),))
            conn.commit()

    def redeem_reset(self, raw, encoded, bootstrap, email):
        if not isinstance(raw, str) or not TOKEN.fullmatch(raw) or not valid_password_hash(encoded):
            return False
        with closing(self.connect()) as conn:
            # Lock the single account so concurrent links cannot both succeed.
            cur, row = self.lock_account(conn)
            current = self.effective(row, bootstrap)
            if not valid_password_hash(current):
                return False
            cur.execute(self.sql("DELETE FROM admin_reset_tokens WHERE id=? AND expires>? AND owner=? RETURNING id"),
                        (fingerprint(raw), int(time.time()), fingerprint(current + '\n' + email)))
            if not cur.fetchone():
                return False
            cur.execute(self.sql("UPDATE admin_credentials SET bootstrap=?,password_hash=? WHERE id=1"),
                        (fingerprint(bootstrap), encoded))
            cur.execute("DELETE FROM admin_access_tokens")
            cur.execute("DELETE FROM admin_sessions")
            cur.execute("DELETE FROM admin_reset_tokens")
            conn.commit()
        return True

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
        return store.credential(os.getenv("VAKSTAAL_ADMIN_PASSWORD_HASH", ""))

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
            try:
                encoded = await run_in_threadpool(credential)
            except Exception:
                LOG.error("Credential validation unavailable")
                return reply(503, "Sessiecontrole tijdelijk niet beschikbaar. Probeer opnieuw.")
            if not valid_password_hash(encoded):
                return reply(503, "Beveiliging is nog niet ingesteld. Neem contact op met de beheerder.")
            if request.method not in {"GET", "HEAD"} and request.headers.get("origin") not in origins:
                return reply(403, "Dit verzoek komt niet van het vaste calculatoradres.")
            p = request.url.path
            if p in {"/api/auth/login", "/api/auth/session", "/api/auth/logout", "/api/auth/forgot", "/api/auth/reset"}:
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
        encoded = await run_in_threadpool(credential)
        password_ok = await run_in_threadpool(verify_password, data["password"], encoded)
        supplied_email = data.get("email", "")
        email_ok = isinstance(supplied_email, str) and hmac.compare_digest(
            utf8(supplied_email.strip().lower()), utf8(admin_email()))
        if not password_ok or not email_ok:
            return reply(401, "E-mailadres of wachtwoord is niet juist.")
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

    async def reset_body(request):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > 4096:
                return None
        try:
            data = json.loads(body)
            return data if isinstance(data, dict) else None
        except (ValueError, UnicodeError):
            return None

    def deliver_reset(supplied_email):
        # Runs after the generic response, including for unknown addresses.
        # Never log the message, SMTP exception, password, or reset token.
        raw = None
        try:
            email = admin_email()
            if not hmac.compare_digest(supplied_email.encode(), email.encode()):
                return
            if not store.limited("reset-mail", 3, RESET_SECONDS):
                return
            origin = next(iter(origins))
            parsed = urlsplit(origin)
            if parsed.scheme != "https" or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
                raise ValueError("Invalid configured origin")
            raw = store.new_reset(os.getenv("VAKSTAAL_ADMIN_PASSWORD_HASH", ""), email)
            if raw is None:
                return
            # Fragment never reaches web-server access logs. Browser removes it
            # immediately and sends the token only in the POST request body.
            link = origin + "/reset-password#token=" + raw
            send_auth_mail("Stel je Vakstaal-wachtwoord opnieuw in",
                           "Je hebt een nieuw wachtwoord aangevraagd voor de Vakstaal calculator.\n\n"
                           + link + "\n\nDeze link is 15 minuten geldig en werkt één keer.\n"
                           "Heb je dit niet aangevraagd? Negeer deze mail; je wachtwoord blijft hetzelfde.\n")
        except Exception:
            LOG.error("Password reset email could not be sent; check SMTP configuration")
            if raw:
                try:
                    store.discard_reset(raw)
                except Exception:
                    LOG.error("Failed reset token cleanup unavailable")

    @app.post("/api/auth/forgot")
    async def forgot(request: Request, background_tasks: BackgroundTasks):
        data = await reset_body(request)
        if not data or not isinstance(data.get("email"), str) or len(data["email"]) > 254 or not utf8(data["email"]):
            return reply(400, "Vul een geldig e-mailadres in.")
        if not await run_in_threadpool(store.limited, "forgot", 20, 300):
            return reply(429, "Te veel aanvragen. Wacht vijf minuten en probeer opnieuw.")
        background_tasks.add_task(deliver_reset, data["email"].strip().lower())
        return reply(200, "Als dit e-mailadres bij het beheerdersaccount hoort, ontvang je een herstellink. Controleer ook je spammap. Maximaal drie herstelmails per 15 minuten.")

    @app.post("/api/auth/reset")
    async def reset(request: Request, background_tasks: BackgroundTasks):
        data = await reset_body(request)
        if not data:
            return reply(400, "Ongeldig herstelverzoek.")
        if not await run_in_threadpool(store.limited, "reset", 8, 300):
            return reply(429, "Te veel pogingen. Wacht vijf minuten en probeer opnieuw.")
        password, confirm, raw = data.get("password"), data.get("confirm_password"), data.get("token")
        if not isinstance(password, str) or not 15 <= len(password) <= 128 or not 1 <= len(utf8(password)) <= 512:
            return reply(400, "Gebruik een wachtwoord van 15 tot 128 tekens.")
        if password != confirm:
            return reply(400, "De wachtwoorden zijn niet gelijk.")
        if not isinstance(raw, str) or not TOKEN.fullmatch(raw):
            return reply(400, "Deze herstellink is ongeldig of verlopen. Vraag een nieuwe aan.")
        encoded = await run_in_threadpool(hash_password, password)
        changed = await run_in_threadpool(store.redeem_reset, raw, encoded,
                                         os.getenv("VAKSTAAL_ADMIN_PASSWORD_HASH", ""), admin_email())
        if not changed:
            return reply(400, "Deze herstellink is ongeldig of verlopen. Vraag een nieuwe aan.")
        background_tasks.add_task(send_change_notice)
        response = reply(200, "Je wachtwoord is gewijzigd. Log opnieuw in met je nieuwe wachtwoord.")
        response.delete_cookie(COOKIE, path="/", secure=True, httponly=True, samesite="strict")
        return response

    return store
