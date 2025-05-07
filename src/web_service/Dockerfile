FROM doctor-base:latest AS web_service

# The WORKDIR, ENV, and code are inherited from the base image.
# If specific overrides or additional ENV vars are needed for web_service, set them here.

# Expose the port
EXPOSE 9111

# Run the web service
CMD ["python", "src/web_service/main.py"]
