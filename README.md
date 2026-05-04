# API de análisis de iniciativas (Iniciativa AI)

Microservicio **FastAPI** con MariaDB externo (historial y conversaciones) y Qdrant (vectores). Está diseñado para desplegarse **por separado** de la aplicación Yii2 *Talento humano*; la web solo hace de proxy hacia este servicio e inyecta el usuario con el encabezado `X-User-Id`.

## Contenido

1. [Estructura del código](#estructura-del-código)
2. [Inicio rápido (Docker)](#inicio-rápido-docker)
3. [Variables de entorno](#variables-de-entorno)
4. [Desarrollo sin Docker](#desarrollo-sin-docker)
5. [Integración con el front Yii](#integración-con-el-front-yii)
6. [API y documentación interactiva](#api-y-documentación-interactiva)
7. [Migración a un repositorio Git nuevo](#migración-a-un-repositorio-git-nuevo)

---

## Estructura del código

El paquete principal es **`app/`**. La raíz expone `main.py` como envoltorio para que sigan valiendo `uvicorn main:app` y el `Dockerfile` actual.

| Ruta | Rol |
|------|-----|
| `main.py` | Reexporta `app` desde `app.main` (compatibilidad con uvicorn/Docker). |
| `app/main.py` | Instancia FastAPI, CORS, evento de arranque, registro de routers. |
| `app/api/routers/` | Rutas HTTP (`health`, `conversations`, `history`). |
| `app/api/deps.py` | Dependencias compartidas (`X-User-Id`, propiedad de conversación, fechas). |
| `app/schemas/` | Modelos Pydantic de entrada/salida de la API. |
| `app/db.py` | SQLAlchemy: modelos, sesión, `init_db`. |
| `app/agent_logic.py` | Llamadas a LLM, RAG y herramientas de formulario. |
| `app/qdrant_client_setup.py` | Cliente y colecciones Qdrant. |

También puedes arrancar con `uvicorn app.main:app` si prefieres referenciar el módulo explícitamente.

## Inicio rápido (Docker)

```bash
cd ai-backend
cp .env.example .env
# Editar .env: OPENAI_API_KEY y DATABASE_URL obligatorios
docker compose up -d
```

- API en el host: `http://localhost:8008` (puedes cambiar el puerto con `API_PORT` en `.env`).

## Variables de entorno

| Variable | Descripción |
|----------|-------------|
| `DATABASE_URL` | Cadena SQLAlchemy para MariaDB, p. ej. `mariadb+pymysql://user:pass@host.docker.internal:3306/talento_humano` |
| `QDRANT_URL` | Base URL de Qdrant, p. ej. `http://qdrant:6333` |
| `OPENAI_API_KEY` | Clave de OpenAI |

Detalle adicional en `.env.example`.

## Desarrollo sin Docker

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows; en Unix: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Integración con el front Yii

En el servidor o en Docker donde corre PHP:

- `AI_BACKEND_INTERNAL_URL` = URL **interna** con la que PHP puede alcanzar este API, por ejemplo:
  - Misma máquina, PHP en contenedor: `http://host.docker.internal:8008` (puerto mapeado por defecto de este `docker compose`).
  - Misma red Docker: `http://<nombre-servicio-api>:8000`.
  - Producción: URL interna o del balanceador hacia el microservicio.

No es necesario que el navegador abra el API: el módulo Iniciativa usa un proxy en Yii; el `user_id` **no** debe confiar en el cliente, solo en el encabezado que añade el proxy.

Para lecturas de comité/TI, el proxy debe enviar `X-Th-Rbac-Roles` con roles separados por coma, pipe o espacios. Los endpoints `GET /history/{id}`, `GET /workflow/{id}` y `GET /workflow/{id}/timeline` permiten acceso si el usuario es dueño de la conversación o si el header contiene `TI`/`ti` o `iniciativa_comite`.

Las mutaciones de comité (`POST .../request-technical-review`, `POST .../committee-responses`, `POST .../submit-final-committee`) usan la misma regla: dueño **o** rol `iniciativa_comite` en `X-Th-Rbac-Roles`. La evaluación TI (`POST .../technical-evaluations`) permite dueño **o** rol `ti`/`TI` en ese header. `POST .../submit-committee` sigue siendo solo para el dueño (envío a comité inicial).

## API y documentación interactiva

Con el servicio en marcha, FastAPI expone:

- **Swagger UI:** `GET /docs`
- **ReDoc:** `GET /redoc`

### Endpoints (resumen)

| Método | Ruta | Notas |
|--------|------|--------|
| `GET` | `/health` | Sin dependencia de BD |
| `POST` | `/conversations`, `/analyze`, `/chat` | Lógica principal |
| `GET` | `/history`, `/history/{id}` | Requieren `X-User-Id` (proxy Yii) |
| `POST` | `/workflow/{id}/submit-committee` | Envía una iniciativa analizada a comité inicial |
| `POST` | `/workflow/{id}/request-technical-review` | Solicita evaluación técnica de TI |
| `POST` | `/workflow/{id}/technical-evaluations` | Registra rúbrica y comentarios de TI |
| `POST` | `/workflow/{id}/committee-responses` | Registra revisión/respuesta posterior de comité u otro usuario |
| `POST` | `/workflow/{id}/submit-final-committee` | Envía la iniciativa a comité final |
| `GET` | `/workflow` | Lista workflows por estado; default `committee_review` para bandeja de comité |
| `GET` | `/workflow/{id}`, `/workflow/{id}/timeline` | Consulta estado actual, evaluaciones y línea de tiempo |

Para detalle de cuerpos de petición y respuestas, usa `/docs` o el código bajo `app/api/routers/` y `app/schemas/`.

### Workflow híbrido de iniciativas

El workflow de comité/TI vive en este microservicio como registro de estado y timeline, ligado a la conversación por `conversation_id`. La app Yii `talento_humano` sigue siendo responsable de pantallas, permisos y validación de roles; el proxy debe inyectar `X-User-Id` y puede enviar `actor_role`, `actor_name` y `payload` para enriquecer los eventos.

Estados usados por el API:

- `draft_analyzed`: iniciativa generada y analizada, lista para enviarse a comité.
- `committee_review`: en comité inicial.
- `technical_review`: enviada a miembros de TI para evaluación.
- `committee_with_technical_feedback`: vuelve a comité con comentarios y rúbrica TI.
- `user_reviewed`: otro usuario/comité respondió la iniciativa.
- `final_committee`: enviada a comité final.

La evaluación técnica se registra con una rúbrica de criterios puntuados de 1 a 5. El API calcula `total_score`, `average_score` y `complexity` (`baja`, `media`, `alta`) y también escribe un evento en `/workflow/{id}/timeline`.

Para la bandeja de comité, usa `GET /workflow`. Por defecto devuelve iniciativas en `committee_review` del usuario actual. Si Yii ya validó que el usuario puede ver la bandeja global, debe enviar `X-Th-Rbac-Roles: iniciativa_comite` o `X-Th-Rbac-Roles: TI` y llamar `GET /workflow?owned_only=false`. Para incluir varios estados, envía el parámetro repetido, por ejemplo `GET /workflow?owned_only=false&statuses=committee_review&statuses=committee_with_technical_feedback`.

`POST /workflow/{id}/committee-responses` acepta `response_type: "committee_initial_comment"` para registrar una nota del comité inicial en el timeline sin cambiar el estado `committee_review`. Otros `response_type` continúan avanzando el workflow a `user_reviewed`.

Ejemplo mínimo para evaluación TI:

```json
{
  "evaluator_name": "María TI",
  "actor_role": "ti",
  "comment": "Requiere integración con ERP y revisión de seguridad.",
  "criteria": [
    { "criterion": "integraciones", "score": 4, "comment": "ERP y SSO" },
    { "criterion": "datos", "score": 3 },
    { "criterion": "seguridad", "score": 4 },
    { "criterion": "infraestructura", "score": 3 },
    { "criterion": "esfuerzo", "score": 4 }
  ]
}
```

## Migración a un repositorio Git nuevo

1. **Copia limpia** (simple): copia todo el directorio de este servicio a una carpeta nueva, añade `README.md` y `docker-compose.yml` ya incluidos, y:

   ```bash
   git init
   git add .
   git commit -m "Initial: microservicio Iniciativa AI"
   git remote add origin <url-nuevo-repositorio>
   git push -u origin main
   ```

2. **Mantener historial** desde un monorepo: en el repositorio que contiene la carpeta `ai-backend/`, desde la raíz de ese monorepo:

   ```bash
   git subtree split -P ai-backend -b rama-solo-iniciativa-ai
   ```

   Luego clona o crea un repo vacío, y haz `git pull` de esa rama, o añade el remoto y fusiona. (Si renombraste la carpeta, ajusta `-P` al nombre real.)

3. Tras publicar el microservicio, en el repositorio de la app Yii elimina o deja de actualizar el código duplicado de `ai-backend` y ajusta solo `AI_BACKEND_INTERNAL_URL` y el despliegue.

---

### Cómo escalar esta documentación

- **README:** propósito del servicio, cómo levantarlo y enlaces a detalle.
- **Detalle largo** (runbooks, decisiones de arquitectura, ADRs): mejor en `docs/` con enlaces desde aquí cuando aparezcan.
- **Contrato HTTP:** la fuente de verdad puede seguir siendo OpenAPI (`/openapi.json`); el resumen en tabla solo complementa.
