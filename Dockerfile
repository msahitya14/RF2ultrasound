FROM node:20-bullseye-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    iproute2 \
    net-tools \
    lsof \
    libnss3-tools \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Install mkcert manually
# Install mkcert manually and generate generic certificates
RUN curl -L https://github.com/FiloSottile/mkcert/releases/download/v1.4.4/mkcert-v1.4.4-linux-amd64 -o /usr/local/bin/mkcert \
    && chmod +x /usr/local/bin/mkcert \
    && /usr/local/bin/mkcert -install \
    && /usr/local/bin/mkcert -key-file key.pem -cert-file cert.pem localhost 127.0.0.1 0.0.0.0

# Copy package files
COPY package.json package-lock.json* ./

# Install Node dependencies
RUN npm install

# Copy everything else
COPY . .

# Expose ports for Backend and Vite
EXPOSE 3000
EXPOSE 5173

# Run the app
CMD ["sh", "-c", "node server.cjs & npm run dev -- --host"]