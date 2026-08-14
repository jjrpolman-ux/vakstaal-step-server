# Vakstaal STEP Server – Render

Deze repository bevat alleen de zware STEP/CadQuery-backend voor de Vakstaal offertecalculator.

## Render

Maak in Render een nieuwe Web Service vanuit deze GitHub-repository.

Aanbevolen:
- Language/Runtime: Docker
- Region: Frankfurt (of dichtstbijzijnde EU-regio)
- Branch: main
- Dockerfile: ./Dockerfile
- Health Check Path: /health
- Instance: Starter of groter

Render verwacht dat de server bindt aan `0.0.0.0` en de `PORT`-omgevingvariabele.
Deze server doet dat automatisch.

Na een succesvolle deploy krijg je een URL zoals:

https://vakstaal-step-server.onrender.com

Test vervolgens:

https://vakstaal-step-server.onrender.com/health

Daar moet `{"ok":true,...}` verschijnen.

## API

POST `/api/analyze-step`
- multipart veld: `file`
- retourneert STEP-details plus `job_id`

GET `/api/solid-mesh/{job_id}/{solid_index}`
- retourneert de 3D mesh voor één solid

STEP-bestanden worden tijdelijk in `/tmp` bewaard en standaard na 6 uur opgeruimd.

## Volgende stap

Na Render deploy wordt de publieke Render-URL in de Vakstaal Vercel/iPhone-app gezet.
