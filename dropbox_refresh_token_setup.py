import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from getpass import getpass


TOKEN_URL = "https://api.dropboxapi.com/oauth2/token"


def ask_nonempty(label: str, *, secret: bool = False) -> str:
    while True:
        value = getpass(label) if secret else input(label)
        value = value.strip()
        if value:
            return value
        print("Dit veld mag niet leeg zijn.\n")


def main():
    print("=" * 62)
    print("Vakstaal - Dropbox refresh token instellen")
    print("=" * 62)
    print()
    print("Gebruik gegevens van DEZELFDE zakelijke Dropbox Developer-app.")
    print("De App secret en authorization code worden niet zichtbaar tijdens het typen.")
    print()

    app_key = ask_nonempty("App key: ")
    app_secret = ask_nonempty("App secret: ", secret=True)
    auth_code = ask_nonempty("Nieuwe authorization code: ", secret=True)

    payload = urllib.parse.urlencode({
        "code": auth_code,
        "grant_type": "authorization_code",
        "client_id": app_key,
        "client_secret": app_secret,
    }).encode("utf-8")

    request = urllib.request.Request(
        TOKEN_URL,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "Vakstaal-Dropbox-Setup/1.0",
        },
    )

    print("\nContact maken met Dropbox...")

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        print(f"\nDropbox gaf HTTP {exc.code}.")
        try:
            err = json.loads(raw)
            print("Foutmelding:", err.get("error_description") or err.get("error") or raw)
        except Exception:
            print("Foutmelding:", raw)
        print()
        print("Controleer:")
        print("- of de authorization code NIEUW is en nog niet eerder gebruikt;")
        print("- of App key en App secret van exact dezelfde Dropbox-app zijn;")
        print("- of de autorisatielink token_access_type=offline bevatte.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nVerbinding mislukt: {type(exc).__name__}: {exc}")
        sys.exit(1)

    refresh_token = result.get("refresh_token")
    access_token = result.get("access_token")

    if not refresh_token:
        print("\nDropbox gaf wel een antwoord, maar GEEN refresh_token.")
        print("Waarschijnlijk is de authorization code niet met offline access aangemaakt.")
        print("Maak een nieuwe code met token_access_type=offline en probeer opnieuw.")
        sys.exit(2)

    print("\n" + "=" * 62)
    print("GESLAAGD")
    print("=" * 62)
    print()
    print("Kopieer onderstaande waarde direct naar Render.")
    print("Deel hem niet in chat of screenshots.")
    print()
    print("DROPBOX_REFRESH_TOKEN=")
    print(refresh_token)
    print()
    print("Voeg daarnaast in Render toe:")
    print("DROPBOX_APP_KEY=" + app_key)
    print("DROPBOX_APP_SECRET=<jouw App secret>")
    print()
    print("De bestaande DROPBOX_ACCESS_TOKEN mag voorlopig blijven staan")
    print("totdat de servercode is aangepast voor automatische tokenvernieuwing.")
    print()
    if access_token:
        print("Dropbox heeft ook een tijdelijke access token teruggegeven;")
        print("die hoef je voor deze stap niet te kopiëren.")


if __name__ == "__main__":
    main()
