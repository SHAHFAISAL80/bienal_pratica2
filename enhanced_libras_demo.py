# setup.py - Automated setup script for Bienal Prática Project
"""
This script automates the setup process for the WebAR Museum project
Run: python setup.py
"""

import os
import sys
import json
import subprocess
import sqlite3
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required Python packages"""
    requirements = [
        "flask>=2.0.0",
        "flask-cors>=4.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "google-cloud-vision>=3.0.0",
        "google-cloud-texttospeech>=2.14.0",
        "google-generativeai>=0.3.0",
        "Pillow>=8.0.0",
        "requests>=2.25.0"
    ]
    
    print("📦 Installing Python packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "artworks",
        "static/audio",
        "uploads",
        "logs",
        "config",
        "tests"
    ]
    
    print("📁 Creating directory structure...")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created {directory}/")
    
    return True

def create_config_files():
    """Create configuration files"""
    
    # config/settings.json
    config_data = {
        "google_cloud": {
            "credentials_path": "config/google-cloud-credentials.json",
            "project_id": "your-project-id"
        },
        "gemini": {
            "api_key": "your-gemini-api-key",
            "model": "gemini-1.5-flash"
        },
        "database": {
            "path": "museum.db"
        },
        "server": {
            "host": "0.0.0.0",
            "port": 5000,
            "debug": True
        },
        "performance": {
            "confidence_threshold": 0.7,
            "max_response_time": 2.0,
            "cache_enabled": True
        },
        "languages": {
            "supported": ["pt", "en", "es"],
            "default": "pt"
        }
    }
    
    with open("config/settings.json", "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)
    print("✅ Created config/settings.json")
    
    # Create sample artwork data CSV
    sample_artworks_csv = """id,name,artist,year,description,image_path
1,Abaporu,Tarsila do Amaral,1928,"Pintura modernista icônica do movimento antropófago brasileiro",artworks/abaporu.jpg
2,O Mestiço,Candido Portinari,1934,"Representação da diversidade racial brasileira",artworks/mestizo.jpg
3,Saudade,Almeida Júnior,1899,"Retrato melancólico de uma figura feminina",artworks/saudade.jpg
4,O Lavrador de Café,Candido Portinari,1939,"Retrato do trabalhador rural brasileiro",artworks/lavrador.jpg
5,Tropical,Anita Malfatti,1917,"Obra expressionista com cores vibrantes",artworks/tropical.jpg
6,Pau-Brasil,Tarsila do Amaral,1925,"Representação da natureza brasileira",artworks/pau_brasil.jpg
7,O Violeiro,Almeida Júnior,1899,"Cena do cotidiano caipira paulista",artworks/violeiro.jpg
8,Operários,Tarsila do Amaral,1933,"Crítica social sobre a industrialização",artworks/operarios.jpg
9,Menino com Lagartixa,Lasar Segall,1924,"Retrato infantil com elementos simbólicos",artworks/menino_lagartixa.jpg
10,São Paulo,Tarsila do Amaral,1924,"Vista urbana da cidade em crescimento",artworks/sao_paulo.jpg
11,O Vendedor de Frutas,Tarsila do Amaral,1925,"Cena de mercado popular",artworks/vendedor_frutas.jpg
12,Autorretrato,Lasar Segall,1927,"Autorretrato expressionista",artworks/autorretrato.jpg
13,Café,Candido Portinari,1935,"Painéis sobre a cultura cafeeira",artworks/cafe.jpg
14,O Homem Amarelo,Anita Malfatti,1917,"Figura expressionista marcante",artworks/homem_amarelo.jpg
15,Urutu,Tarsila do Amaral,1928,"Paisagem brasileira estilizada",artworks/urutu.jpg
16,Bananeiras,Lasar Segall,1927,"Paisagem tropical brasileira",artworks/bananeiras.jpg
17,A Negra,Tarsila do Amaral,1923,"Figura feminina afro-brasileira",artworks/negra.jpg
18,Morro da Favela,Tarsila do Amaral,1924,"Primeira representação de favela na arte",artworks/morro_favela.jpg
19,O Mamoeiro,Tarsila do Amaral,1925,"Natureza morta tropical",artworks/mamoeiro.jpg
20,Guerra,Lasar Segall,1942,"Crítica aos horrores da guerra",artworks/guerra.jpg"""
    
    with open("artworks_data.csv", "w", encoding="utf-8") as f:
        f.write(sample_artworks_csv)
    print("✅ Created artworks_data.csv")
    
    return True

def create_test_files():
    """Create test files"""
    
    # test_recognition.py
    test_code = '''# test_recognition.py - Test script for artwork recognition
import requests
import os

def test_recognition_api():
    """Test the recognition API with sample images"""
    
    api_url = "http://localhost:5000/api/recognize"
    
    # Test with sample images
    test_images = [
        "tests/test_image1.jpg",
        "tests/test_image2.jpg"
    ]
    
    for image_path in test_images:
        if os.path.exists(image_path):
            print(f"Testing with {image_path}...")
            
            with open(image_path, 'rb') as f:
                files = {'image': f}
                response = requests.post(api_url, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"✅ Recognition result: {result}")
                else:
                    print(f"❌ API Error: {response.status_code}")
        else:
            print(f"⚠️ Test image not found: {image_path}")

def test_chat_api():
    """Test the chat API"""
    
    api_url = "http://localhost:5000/api/chat"
    
    test_questions = [
        {"question": "Quem pintou o Abaporu?", "language": "pt"},
        {"question": "What is modernism in Brazilian art?", "language": "en"},
        {"question": "¿Cuándo se pintó esta obra?", "language": "es", "artwork_id": 1}
    ]
    
    for test in test_questions:
        print(f"Testing question: {test['question']}")
        
        response = requests.post(api_url, json=test)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat response: {result.get('text', 'No text')}")
        else:
            print(f"❌ Chat API Error: {response.status_code}")

if __name__ == "__main__":
    print("🧪 Testing Museum API...")
    print("Make sure the server is running (python museum_app.py)")
    
    test_recognition_api()
    test_chat_api()
'''
    
    with open("tests/test_system.py", "w", encoding="utf-8") as f:
        f.write(test_code)
    print("✅ Created tests/test_system.py")
    
    return True

def create_docker_files():
    """Create Docker configuration files"""
    
    # Dockerfile
    dockerfile = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libsm6 \\
    libxext6 \\
    libxrender-dev \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p static/audio uploads logs

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "museum_app.py"]
'''
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    print("✅ Created Dockerfile")
    
    # docker-compose.yml
    docker_compose = '''version: '3.8'

services:
  museum-api:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./artworks:/app/artworks:ro
      - ./config:/app/config:ro
      - ./static:/app/static
      - ./logs:/app/logs
    environment:
      - FLASK_ENV=production
      - GOOGLE_APPLICATION_CREDENTIALS=/app/config/google-cloud-credentials.json
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - museum-api
    restart: unless-stopped
'''
    
    with open("docker-compose.yml", "w") as f:
        f.write(docker_compose)
    print("✅ Created docker-compose.yml")
    
    return True

def create_requirements_file():
    """Create requirements.txt"""
    requirements = """flask>=2.0.0
flask-cors>=4.0.0
opencv-python>=4.5.0
numpy>=1.21.0
google-cloud-vision>=3.0.0
google-cloud-texttospeech>=2.14.0
google-generativeai>=0.3.0
Pillow>=8.0.0
requests>=2.25.0
gunicorn>=20.1.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✅ Created requirements.txt")
    
    return True

def setup_database():
    """Initialize the database with sample data"""
    print("🗄️ Setting up database...")
    
    # This would normally be done by the main application
    # For now, we create the basic structure
    
    try:
        conn = sqlite3.connect("museum.db")
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS artworks (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                artist TEXT NOT NULL,
                year TEXT,
                description TEXT,
                image_path TEXT,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def display_final_instructions():
    """Display final setup instructions"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE! 🎉")
    print("="*60)
    
    print("\n📋 NEXT STEPS:")
    print("\n1. Configure API Keys:")
    print("   • Edit config/settings.json")
    print("   • Add your Google Cloud credentials to config/")
    print("   • Add your Gemini API key")
    
    print("\n2. Add Artwork Images:")
    print("   • Place 20 artwork images in artworks/ folder")
    print("   • Ensure filenames match the CSV data")
    print("   • Supported formats: JPG, PNG")
    
    print("\n3. Run the Application:")
    print("   • python museum_app.py")
    print("   • Open http://localhost:5000/api/health")
    
    print("\n4. Test the System:")
    print("   • python tests/test_system.py")
    print("   • Upload test images via API")
    print("   • Try chat functionality")
    
    print("\n5. Production Deployment:")
    print("   • Use Docker: docker-compose up")
    print("   • Or use Gunicorn: gunicorn museum_app:app")
    
    print("\n📚 API ENDPOINTS:")
    print("   • POST /api/recognize - Upload image for recognition")
    print("   • POST /api/chat - Chat with museum assistant")
    print("   • GET /api/artworks - List all artworks")
    print("   • GET /api/health - System health check")
    
    print("\n🔧 TROUBLESHOOTING:")
    print("   • Check logs/ folder for error logs")
    print("   • Verify API keys are correctly set")
    print("   • Ensure all dependencies are installed")
    print("   • Test with curl commands first")
    
    print("\n💡 TIPS:")
    print("   • Start with a few test images")
    print("   • Monitor performance with /api/health")
    print("   • Use high-quality artwork images (min 800x600)")
    print("   • Test different lighting conditions")
    print("   • Cache frequently asked questions")
    
    print("\n" + "="*60)

def main():
    """Main setup function"""
    print("🎨 Bienal Prática WebAR Museum Setup")
    print("=" * 40)
    
    # Check system requirements
    check_python_version()
    
    # Create directory structure
    if not create_directory_structure():
        print("❌ Failed to create directories")
        return False
    
    # Create configuration files
    if not create_config_files():
        print("❌ Failed to create config files")
        return False
    
    # Create requirements file
    if not create_requirements_file():
        print("❌ Failed to create requirements file")
        return False
    
    # Install Python packages
    install_choice = input("\n📦 Install Python packages now? (y/n): ")
    if install_choice.lower() == 'y':
        if not install_requirements():
            print("❌ Failed to install some packages")
            return False
    
    # Create test files
    if not create_test_files():
        print("❌ Failed to create test files")
        return False
    
    # Create Docker files
    docker_choice = input("\n🐳 Create Docker configuration? (y/n): ")
    if docker_choice.lower() == 'y':
        if not create_docker_files():
            print("❌ Failed to create Docker files")
            return False
    
    # Setup database
    if not setup_database():
        print("❌ Failed to setup database")
        return False
    
    # Display final instructions
    display_final_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("✨ Setup completed successfully!")
    else:
        print("❌ Setup failed. Please check the errors above.")
        sys.exit(1)


# ==========================================
# Additional Configuration Templates
# ==========================================

# nginx.conf - Nginx configuration for production
nginx_config = '''events {
    worker_connections 1024;
}

http {
    upstream museum_app {
        server museum-api:5000;
    }
    
    server {
        listen 80;
        client_max_body_size 50M;
        
        location / {
            proxy_pass http://museum_app;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
        
        location /static/ {
            alias /app/static/;
            expires 1d;
            add_header Cache-Control "public, immutable";
        }
    }
}'''

# systemd service file for Linux deployment
systemd_service = '''[Unit]
Description=Bienal Prática WebAR Museum API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/museum
ExecStart=/opt/museum/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 4 museum_app:app
Restart=always
RestartSec=10
Environment=PYTHONPATH=/opt/museum
Environment=GOOGLE_APPLICATION_CREDENTIALS=/opt/museum/config/credentials.json

[Install]
WantedBy=multi-user.target'''

# Environment variables template
env_template = '''# Environment Variables for Bienal Prática Project
# Copy this to .env and fill in your actual values

# Google Cloud Configuration
GOOGLE_APPLICATION_CREDENTIALS=config/google-cloud-credentials.json
GOOGLE_CLOUD_PROJECT_ID=your-project-id

# Gemini AI Configuration
GEMINI_API_KEY=your-gemini-api-key-here

# Database Configuration
DATABASE_URL=sqlite:///museum.db

# Server Configuration
FLASK_ENV=development
FLASK_DEBUG=True
HOST=0.0.0.0
PORT=5000

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-here

# Performance Settings
CONFIDENCE_THRESHOLD=0.7
MAX_RESPONSE_TIME=2.0
CACHE_ENABLED=True
MAX_UPLOAD_SIZE=50MB

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/museum.log

# Rate Limiting
RATE_LIMIT_RECOGNITION=10/minute
RATE_LIMIT_CHAT=30/minute
RATE_LIMIT_API=60/minute'''

def create_additional_configs():
    """Create additional configuration files"""
    
    # Create nginx config
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    print("✅ Created nginx.conf")
    
    # Create systemd service
    with open("museum.service", "w") as f:
        f.write(systemd_service)
    print("✅ Created museum.service")
    
    # Create environment template
    with open(".env.template", "w") as f:
        f.write(env_template)
    print("✅ Created .env.template")

# Shell scripts for deployment
deploy_script = '''#!/bin/bash
# deploy.sh - Deployment script for production

echo "🚀 Deploying Bienal Prática Museum..."

# Stop existing service
sudo systemctl stop museum || true

# Backup current deployment
if [ -d "/opt/museum" ]; then
    sudo cp -r /opt/museum /opt/museum.backup.$(date +%Y%m%d_%H%M%S)
fi

# Create deployment directory
sudo mkdir -p /opt/museum
sudo chown $USER:$USER /opt/museum

# Copy files
cp -r * /opt/museum/
cd /opt/museum

# Setup Python virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup database
python -c "from museum_app import DatabaseManager; DatabaseManager().init_database()"

# Import artwork data
python -c "
from museum_app import DatabaseManager, DatasetManager
db = DatabaseManager()
dataset = DatasetManager(db)
dataset.import_artworks_from_csv('artworks_data.csv')
"

# Setup systemd service
sudo cp museum.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable museum
sudo systemctl start museum

# Setup nginx
sudo cp nginx.conf /etc/nginx/sites-available/museum
sudo ln -sf /etc/nginx/sites-available/museum /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx

echo "✅ Deployment complete!"
echo "🌐 Service available at: http://your-domain.com"
echo "📊 Check status: sudo systemctl status museum"
'''

# Development start script
dev_script = '''#!/bin/bash
# start_dev.sh - Start development server

echo "🔧 Starting Bienal Prática Museum (Development Mode)"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check configuration
if [ ! -f "config/settings.json" ]; then
    echo "⚠️  Configuration not found. Run setup.py first!"
    exit 1
fi

# Start development server
echo "🚀 Starting development server..."
export FLASK_ENV=development
export FLASK_DEBUG=True
python museum_app.py
'''

def create_shell_scripts():
    """Create shell scripts for deployment and development"""
    
    with open("deploy.sh", "w") as f:
        f.write(deploy_script)
    os.chmod("deploy.sh", 0o755)
    print("✅ Created deploy.sh")
    
    with open("start_dev.sh", "w") as f:
        f.write(dev_script)
    os.chmod("start_dev.sh", 0o755)
    print("✅ Created start_dev.sh")

# Monitoring and health check script
monitoring_script = '''#!/usr/bin/env python3
# monitor.py - System monitoring and health checks

import requests
import time
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class MuseumMonitor:
    def __init__(self, config_file="config/monitoring.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        self.api_url = self.config["api_url"]
        self.check_interval = self.config["check_interval"]
        self.alerts_enabled = self.config["alerts"]["enabled"]
        
    def check_health(self):
        """Check system health"""
        try:
            response = requests.get(f"{self.api_url}/api/health", timeout=30)
            if response.status_code == 200:
                return {"status": "healthy", "data": response.json()}
            else:
                return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def check_recognition_performance(self):
        """Test recognition endpoint performance"""
        try:
            # Use a small test image
            test_image_path = "tests/sample.jpg"
            if os.path.exists(test_image_path):
                start_time = time.time()
                with open(test_image_path, 'rb') as f:
                    files = {'image': f}
                    response = requests.post(f"{self.api_url}/api/recognize", files=files, timeout=30)
                
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    return {"status": "ok", "processing_time": processing_time}
                else:
                    return {"status": "error", "error": f"HTTP {response.status_code}"}
            else:
                return {"status": "skipped", "reason": "No test image available"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def send_alert(self, message):
        """Send alert notification"""
        if not self.alerts_enabled:
            return
        
        try:
            # Email alert (configure SMTP settings in config)
            smtp_config = self.config["alerts"]["email"]
            
            msg = MIMEText(message)
            msg['Subject'] = 'Museum System Alert'
            msg['From'] = smtp_config["from"]
            msg['To'] = smtp_config["to"]
            
            server = smtplib.SMTP(smtp_config["smtp_server"], smtp_config["port"])
            server.starttls()
            server.login(smtp_config["username"], smtp_config["password"])
            server.sendmail(smtp_config["from"], [smtp_config["to"]], msg.as_string())
            server.quit()
            
            print(f"📧 Alert sent: {message}")
            
        except Exception as e:
            print(f"❌ Failed to send alert: {e}")
    
    def run_monitoring(self):
        """Run continuous monitoring"""
        print("🔍 Starting Museum System Monitoring...")
        
        while True:
            try:
                timestamp = datetime.now().isoformat()
                print(f"\\n[{timestamp}] Running health checks...")
                
                # Check overall health
                health = self.check_health()
                print(f"System Health: {health['status']}")
                
                if health["status"] == "unhealthy":
                    self.send_alert(f"System unhealthy: {health.get('error', 'Unknown error')}")
                
                # Check recognition performance
                recognition = self.check_recognition_performance()
                print(f"Recognition: {recognition['status']}")
                
                if recognition["status"] == "ok":
                    processing_time = recognition["processing_time"]
                    print(f"Processing time: {processing_time:.2f}s")
                    
                    if processing_time > 5.0:  # Alert if too slow
                        self.send_alert(f"Recognition performance degraded: {processing_time:.2f}s")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                print("\\n🛑 Monitoring stopped")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    monitor = MuseumMonitor()
    monitor.run_monitoring()
'''

# Create monitoring config
monitoring_config = '''{
    "api_url": "http://localhost:5000",
    "check_interval": 300,
    "alerts": {
        "enabled": false,
        "email": {
            "smtp_server": "smtp.gmail.com",
            "port": 587,
            "username": "your-email@gmail.com",
            "password": "your-app-password",
            "from": "museum-monitor@yoursite.com",
            "to": "admin@yoursite.com"
        }
    },
    "thresholds": {
        "max_response_time": 5.0,
        "min_uptime_percentage": 99.0
    }
}'''

def create_monitoring_files():
    """Create monitoring and maintenance files"""
    
    with open("monitor.py", "w") as f:
        f.write(monitoring_script)
    os.chmod("monitor.py", 0o755)
    print("✅ Created monitor.py")
    
    with open("config/monitoring.json", "w") as f:
        f.write(monitoring_config)
    print("✅ Created config/monitoring.json")