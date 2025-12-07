#!/bin/sh

# 1. Pull the required LLM model
/usr/local/bin/ollama pull llama3:8b

# 2. Pull the required embedding model
/usr/local/bin/ollama pull nomic-embed-text

# 3. Start the server in the foreground
exec /usr/local/bin/ollama serve