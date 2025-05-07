FROM doctor-base:latest AS crawl_worker

# The WORKDIR, ENV, and code are inherited from the base image.
# If specific overrides or additional ENV vars are needed for crawl_worker, set them here.

RUN crawl4ai-setup

# Run the worker
CMD ["python", "-m", "src.crawl_worker.main"]
