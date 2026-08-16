import json
import sys
import urllib.error
import urllib.parse
import urllib.request

TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"

def ask_nonempty(label: str) -> str:
    while True:
        value = input(label).strip()
        if value:
            return value
        print("Dit veld mag niet leeg zijn.\n")

def main():
    print("=" * 62)
    print("Vakstaal - Dropbox refresh token instellen (Render versie)")
    print("=" * 62)
    print()
    print("LET OP: ingevoerde waarden zijn zichtbaar in de Web Shell.")
    print("Maak geen screenshot en deel deze gegevens niet.")
    print()
    print("Gebruik App key, App secret en authorization code")
    print("van exact DEZELFDE zakelijke Dropbox Developer-app.")
    print()

    app_key = ask_nonempty("App key: ")
    app_secret = ask_nonempty("App secret: ")
    auth_code = ask_nonempty("Nieuwe authorization code: ")

    payload = urllib.parse.urlencode({
        "code": auth_code,
        "grant_type": "authorization_code",
        "client_id": app_key,
        "client_secret": app_secret,
    }).encode("utf-8")

    req = urllib.request.Request(
        TOKEN_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Vakstaal-Dropbox-Setup/2.0",
        },
    )

    print("\nContact maken met Dropbox...")

    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        print(f"\nDropbox gaf HTTP {exc.code}.")
        try:
            err = json.loads(raw)
            print("Foutmelding:", err.get("error_description") or err.get("error") or raw)
        except Exception:
            print("Foutmelding:", raw)
        sys.exit(1)
    except Exception as exc:
        print(f"\nVerbinding mislukt: {type(exc).__name__}: {exc}")
        sys.exit(1)

    refresh_token = result.get("refresh_token")
    if not refresh_token:
        print("\nGeen refresh_token ontvangen.")
        print("Maak een nieuwe authorization code via een link met token_access_type=offline.")
        sys.exit(2)

    print("\n" + "=" * 62)
    print("GESLAAGD")
    print("=" * 62)
    print()
    print("Kopieer deze waarde direct naar Render Environment:")
    print()
    print("DROPBOX_REFRESH_TOKEN=")
    print(refresh_token)
    print()
    print("Voeg daarnaast toe:")
    print("DROPBOX_APP_KEY=" + app_key)
    print("DROPBOX_APP_SECRET=<jouw App secret>")
    print()
    print("Laat DROPBOX_ACCESS_TOKEN voorlopig nog staan.")
    print("Daarna wordt server.py aangepast voor automatische vernieuwing.")

if __name__ == "__main__":
    main()
