# FlowScope AI Backend

Python/FastAPI backend for FlowScope AI pipeline platform.

## Setup

### Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env with your settings
```

## Development

### Run with Hot Reload

```bash
# Linux/Mac
./dev.sh

# Windows
dev.bat
```

### Run Production Mode

```bash
python main.py
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Project Structure

```
app/
├── core/              # Configuration, logging
├── pipeline/          # Pipeline engine and storage
├── services/          # Pipeline step implementations
├── routes/            # API endpoints
├── schemas/           # Pydantic models
└── main.py           # Application entry point
```

## Testing API

### Create Pipeline

```bash
curl -X POST http://localhost:8000/pipeline/create \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "name": "Test Pipeline",
      "steps": [
        {"name": "Ingestion", "type": "ingestion", "enabled": true, "config": {}}
      ]
    }
  }'
```

### Run Pipeline

```bash
curl -X POST http://localhost:8000/pipeline/run/{pipeline_id}
```

### Check Status

```bash
curl http://localhost:8000/pipeline/{pipeline_id}/status
```

## Data Storage

Pipeline data is stored in:
```
data/
├── pipelines/         # Pipeline configs and status
└── pipeline_data/     # Stage output data
```
