#!/bin/sh

# 1. Pull the smaller LLM model (phi3:3.8b is roughly ~4GB)
/usr/local/bin/ollama pull phi3:3.8b-mini-4k-instruct-q4_0

# 2. Start the server in the foreground
exec /usr/local/bin/ollama serve