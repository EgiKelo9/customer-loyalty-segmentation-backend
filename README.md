# Customer Loyalty Segmentation Backend API

Aplikasi backend untuk sistem segmentasi loyalitas pelanggan menggunakan machine learning. API ini menyediakan endpoint untuk authentikasi pengguna, analisis data pelanggan, dan clustering berbasis algoritma machine learning.

## 📋 Daftar Isi

- [Teknologi](#-teknologi)
- [Prerequisites](#-prerequisites)
- [Instalasi](#-instalasi)
- [Struktur Folder](#-struktur-folder)
- [Cara Menjalankan](#-cara-menjalankan)
- [API Endpoints](#-api-endpoints)
- [Development](#-development)

## 🛠 Teknologi

| Kategori | Tools |
|----------|-------|
| **Framework** | FastAPI, Uvicorn |
| **Database** | PostgreSQL, SQLAlchemy ORM |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Security** | JWT, PassLib, bcrypt |
| **Visualization** | matplotlib, seaborn |
| **Testing** | pytest, pytest-asyncio |

## 📦 Prerequisites

Pastikan Anda sudah menginstall:

- **Python** 3.10 atau lebih tinggi
- **PostgreSQL** 12 atau lebih tinggi
- **pip** (Python package manager)
- **Git**

### Verifikasi Instalasi

```bash
python --version
psql --version
pip --version
```

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/EgiKelo9/customer-loyalty-segmentation-backend.git
cd customer-loyalty-segmentation-backend
```

### 2. Setup Virtual Environment

```bash
# Membuat virtual environment
python -m venv .venv

# Aktivasi virtual environment
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables

Buat file `.env` di root directory:

```env
# Database Configuration
DATABASE_URL=postgresql://postgres:password@localhost:5432/
DATABASE_NAME=cust_segmentation_db

# Security
SECRET_KEY=your_super_secret_key_here_change_this_in_production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Environment
ENV=dev
```

### 5. Setup Database

```bash
# Jalankan aplikasi (database akan dibuat otomatis)
python -m uvicorn app.main:app --reload
```

Atau setup database secara manual:

```bash
# Buat database dan tables
psql -U postgres -f app/database/init.sql
```

## 📁 Struktur Folder

```
customer-loyalty-segmentation-backend/
│
├── app/                           # Main application directory
│   ├── main.py                    # Entry point FastAPI
│   │
│   ├── controller/                # Business logic & data processing
│   │   ├── auth.py               # Authentication logic
│   │   └── health.py             # System health check logic
│   │
│   ├── core/                      # Core configuration
│   │   └── config.py             # Settings & environment configuration
│   │
│   ├── database/                  # Database configuration
│   │   ├── main.py               # SQLAlchemy setup & session management
│   │   └── init.sql              # Database initialization script
│   │
│   ├── middleware/                # HTTP middleware
│   │   ├── cors.py               # CORS configuration
│   │   └── static.py             # Static files serving
│   │
│   ├── models/                    # SQLAlchemy ORM models
│   │   └── user.py               # User database model
│   │
│   ├── pipeline/                  # Machine learning pipeline
│   │   ├── ml_service.py         # Inference new data
│   │   ├── preprocessing.py      # Data preprocessing & feature engineering
│   │   └── workflow.ipynb        # Jupyter notebook untuk experimentation
│   │
│   ├── router/                    # API route handlers
│   │   ├── auth.py               # Authentication endpoints
│   │   ├── health.py             # Health check endpoints
│   │   └── predict.py            # Prediction endpoints
│   │
│   ├── schemas/                   # Pydantic validation models
│   │   ├── auth.py               # Authentication request/response schemas
│   │   ├── base.py               # Base response schema
│   │   ├── health.py             # Health check schemas
│   │   └── predict.py            # Prediction schemas
│   │
│   └── shared/                    # Shared utilities
│       ├── auth.py               # Auth dependency (Bearer token)
│       ├── token.py              # JWT token utilities
│       └── transaction_manager.py # Database transaction management
│
├── static/                        # Static assets (CSS, JS, images)
│
├── .env                          # Environment variables (create manually)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

### Penjelasan Folder Utama

| Folder | Fungsi |
|--------|--------|
| **app/controller/** | Mengandung business logic dan memproses data dari request |
| **app/core/** | Konfigurasi global aplikasi seperti environment variables |
| **app/database/** | Koneksi database, session management, dan script inisialisasi |
| **app/middleware/** | Middleware untuk CORS, static files, dan logging |
| **app/models/** | Definisi struktur tabel database menggunakan ORM |
| **app/pipeline/** | Pipeline machine learning untuk preprocessing, training, dan prediction |
| **app/router/** | Endpoint API yang menerima request dari client |
| **app/schemas/** | Validasi request/response menggunakan Pydantic |
| **app/shared/** | Fungsi-fungsi utility yang digunakan di berbagai bagian aplikasi |
| **static/** | File statis seperti CSS, JavaScript, images |

## 🏃 Cara Menjalankan

### Mode Development (dengan auto-reload)

```bash
# Pastikan virtual environment sudah aktif
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload
```

### Mode Production

```bash
uvicorn app.main:app --host 0.0.0.0 --port 5000
```

API akan tersedia di: `http://localhost:5000`

### Akses Interactive API Documentation

- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc

## 🔌 API Endpoints

### Authentication

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/api/v1/auth/register` | Register pengguna baru |
| `POST` | `/api/v1/auth/login` | Login dan dapatkan JWT token |

Catatan: Endpoint prediction membutuhkan header `Authorization: Bearer <token>`.

### System Health

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/api/v1/health/` | Check status aplikasi |

### Prediction (Protected)

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/api/v1/predict/lrfm` | Prediksi dari nilai LRFM |
| `POST` | `/api/v1/predict/transactions` | Prediksi dari transaksi JSON (1 pelanggan) |
| `POST` | `/api/v1/predict/transactions/upload` | Prediksi dari file CSV/Excel |

## 💻 Development

### Running Tests

```bash
# Jalankan semua tests
pytest

# Jalankan dengan verbose output
pytest -v

# Jalankan specific test file
pytest app/tests/test_auth.py

# Jalankan dengan coverage report
pytest --cov=app
```

### Code Formatting

```bash
# Format code menggunakan black
black .

# Check code style
flake8 app/
```

### Database Migrations (Jika diperlukan)

```bash
# Buat database baru
python -c "from app.database.main import create_db; create_db()"

# Jalankan manual SQL script
psql -U postgres -d cust_segmentation_db -f app/database/init.sql
```

## 📝 Catatan Penting

1. **JWT Secret Key**: Ganti `SECRET_KEY` di `.env` dengan value yang aman sebelum production
2. **Database Password**: Jangan hardcode password database, gunakan environment variables
3. **CORS Configuration**: Sesuaikan allowed origins di `app/middleware/cors.py` untuk security
4. **Logging**: Setup proper logging untuk production environment

## 🐛 Troubleshooting

### Error: Database does not exist

```bash
# Buat database manual
python -m uvicorn app.main:app --reload
```

### Error: psycopg2 connection failed

Pastikan PostgreSQL running dan credentials di `.env` benar:

```bash
psql -U postgres -h localhost
```

### Port already in use

Gunakan port berbeda:

```bash
uvicorn app.main:app --port 8000
```

## 📄 License

Capstone Project - Pijak: AI for Business Intelligence

## 👥 Tim Pengembang

- Egi Kelo - Backend Development & ML Pipeline
- Baraja Putra - Database & Infrastructure
