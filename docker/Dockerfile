FROM python:3.12-slim

WORKDIR /app
COPY cli /app/cli
COPY examples /app/examples
RUN pip install --no-cache-dir -e /app/cli

ENTRYPOINT ["vision-pipe"]
CMD ["--help"]
