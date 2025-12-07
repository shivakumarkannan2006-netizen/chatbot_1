# Use the official Ollama base image
FROM ollama/ollama:latest

# Set the entrypoint to our custom startup script
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Use the script as the entrypoint
ENTRYPOINT ["/start.sh"]