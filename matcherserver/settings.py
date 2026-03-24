import os
import sys
from pathlib import Path
from dotenv import dotenv_values

BASE_DIR = Path(__file__).resolve().parent.parent

# Ensure project root (BASE_DIR) is on sys.path for `src.*` imports
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

ENV_CONFIG = dotenv_values(os.path.join(BASE_DIR, '.env'))

DEBUG = ENV_CONFIG.get('DEBUG', 'True') == 'True'
SECRET_KEY = ENV_CONFIG.get('SECRET_KEY', 'django-insecure-dev-key')
ALLOWED_HOSTS = ['*']

DATA_UPLOAD_MAX_MEMORY_SIZE = 52428800  # 50MB

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders',
    'matcherapp',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.middleware.gzip.GZipMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'matcherserver.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'matcherserver.wsgi.application'

if ENV_CONFIG.get('USE_SQLITE', 'true').lower() == 'true':
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
            # Longer timeout: background thread scores resumes while the UI polls (avoids "database is locked")
            'OPTIONS': {'timeout': 30},
        }
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.mysql',
            'NAME': ENV_CONFIG.get('DB_NAME', 'resume_matcher'),
            'USER': ENV_CONFIG.get('DB_USER', 'root'),
            'PASSWORD': ENV_CONFIG.get('DB_PASSWORD', ''),
            'HOST': ENV_CONFIG.get('DB_HOST', 'localhost'),
            'PORT': ENV_CONFIG.get('DB_PORT', '3306'),
            'OPTIONS': {'charset': 'utf8mb4'},
        }
    }

AUTH_PASSWORD_VALIDATORS = []
LOGIN_URL = '/login/'
LOGIN_REDIRECT_URL = '/dashboard/'

STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, 'matcherapp', 'static')]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# Background processing via threading.Thread — no Celery/Redis needed

OPENAI_API_KEY = ENV_CONFIG.get('OPENAI_API_KEY', '')
ANTHROPIC_API_KEY = ENV_CONFIG.get('ANTHROPIC_API_KEY', '')
# Always write to os.environ so src/config.py os.environ.get() sees the value
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['ANTHROPIC_API_KEY'] = ANTHROPIC_API_KEY
