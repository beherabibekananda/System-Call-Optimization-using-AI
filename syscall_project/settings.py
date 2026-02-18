"""
Django settings for SysCall AI Project
"""

from pathlib import Path
import os
import dj_database_url
import environ

# Initialize environment variables
env = environ.Env(
    DEBUG=(bool, False),
    ALLOWED_HOSTS=(list, ['*']),
)

BASE_DIR = Path(__file__).resolve().parent.parent

# Read .env file if it exists
environ.Env.read_env(os.path.join(BASE_DIR, '.env'))

SECRET_KEY = env('SECRET_KEY', default='django-insecure-tc^bn+ha0)$3n++1rk)hd3&#2p4%ipafwhmmmdu3y$pmpgw-p7')

DEBUG = env('DEBUG')

ALLOWED_HOSTS = env('ALLOWED_HOSTS')

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "whitenoise.runserver_nostatic",
    # Third-party
    "crispy_forms",
    "crispy_bootstrap5",
    # Our apps
    "accounts",
    "dashboard",
]

CRISPY_ALLOWED_TEMPLATE_PACKS = "bootstrap5"
CRISPY_TEMPLATE_PACK = "bootstrap5"

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "syscall_project.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "syscall_project.wsgi.application"

# Database configuration
db_url = os.environ.get('DATABASE_URL')

if db_url:
    # 1. Fix the hostname if it's .com
    if "supabase.com" in db_url:
        db_url = db_url.replace("supabase.com", "supabase.co")
    
    # 2. Fix unencoded @ in password
    if "@" in db_url.split("://")[-1].split("@")[0]:
        import urllib.parse
        parts = db_url.split("@", 1)
        if len(parts) > 1:
            auth_part = parts[0]
            host_part = parts[1]
            schema_auth = auth_part.split("://")
            if len(schema_auth) > 1:
                schema = schema_auth[0]
                user_pass = schema_auth[1].split(":", 1)
                if len(user_pass) > 1:
                    username = user_pass[0]
                    password = urllib.parse.quote(user_pass[1])
                    db_url = f"{schema}://{username}:{password}@{host_part}"
    
    # Force Django to use this cleaned URL
    DATABASES = {
        'default': dj_database_url.parse(db_url, conn_max_age=600)
    }
else:
    # Fallback to local SQLite if no DATABASE_URL is found
    DATABASES = {
        'default': dj_database_url.config(
            default=f'sqlite:////{os.path.join(BASE_DIR, "db.sqlite3")}',
            conn_max_age=600
        )
    }

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

# Static files
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATICFILES_STORAGE = "whitenoise.storage.CompressedManifestStaticFilesStorage"

# Media (uploaded strace files)
MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# Auth redirects
LOGIN_URL = "/accounts/login/"
LOGIN_REDIRECT_URL = "/dashboard/"
LOGOUT_REDIRECT_URL = "/accounts/login/"

# Default primary key field type
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# File Upload max size (10MB)
FILE_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
DATA_UPLOAD_MAX_MEMORY_SIZE = 10 * 1024 * 1024
