#!/bin/bash
# =============================================================================
# EcoMLOps: EC2 Setup Script
# Run this on a fresh Ubuntu 22.04 EC2 instance to set up the full MLOps stack
# =============================================================================

set -e

echo "============================================================="
echo "  EcoMLOps - EC2 Setup Script"
echo "============================================================="

# Update system
echo "[1/8] Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install Docker
echo "[2/8] Installing Docker..."
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Install Docker Compose v2
echo "[3/8] Installing Docker Compose..."
DOCKER_COMPOSE_VERSION=v2.24.0
sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installations
docker --version
docker-compose --version

# Install AWS CLI
echo "[4/8] Installing AWS CLI..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
sudo apt-get install -y unzip
unzip -q awscliv2.zip
sudo ./aws/install
aws --version
rm -rf awscliv2.zip aws/

# Install Git
echo "[5/8] Installing Git..."
sudo apt-get install -y git

# Create project directory
echo "[6/8] Setting up project directory..."
mkdir -p ~/ecomlops
cd ~/ecomlops

# Clone your repository (replace with your actual repo URL)
# git clone https://github.com/YOUR_USERNAME/ecomlops.git .
# OR if you're uploading files manually, scp them to this directory

echo "[7/8] Creating required directories..."
mkdir -p models metrics

# Set proper permissions
sudo chown -R $USER:$USER ~/ecomlops

echo "[8/8] Setup complete!"
echo ""
echo "============================================================="
echo "  Next Steps:"
echo "============================================================="
echo ""
echo "1. Upload your project files to ~/ecomlops/"
echo "   scp -i your-key.pem -r ./* ubuntu@YOUR_EC2_IP:~/ecomlops/"
echo ""
echo "2. Configure AWS credentials:"
echo "   aws configure"
echo ""
echo "3. Configure GitHub Secrets:"
echo "   - EC2_HOST: Your EC2 public IP"
echo "   - EC2_SSH_KEY: Your .pem file contents"
echo "   - AWS_ACCESS_KEY_ID: Your AWS access key"
echo "   - AWS_SECRET_ACCESS_KEY: Your AWS secret key"
echo "   - ECR_REGISTRY: Your ECR registry URI"
echo ""
echo "4. Start the MLOps stack:"
echo "   cd ~/ecomlops"
echo "   docker-compose up -d"
echo ""
echo "5. Access your services:"
echo "   API:        http://YOUR_EC2_IP:8000"
echo "   MLflow:     http://YOUR_EC2_IP:5000"
echo "   Prometheus: http://YOUR_EC2_IP:9090"
echo "   Grafana:    http://YOUR_EC2_IP:3000 (admin/ecomlops123)"
echo ""
echo "============================================================="
