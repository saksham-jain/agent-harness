FROM python:3.11-slim

# The agent client runs its shell/file tools in the working directory, which
# docker-compose mounts the project onto. The RAG and MCP services pass their
# own paths as arguments, so no service depends on this.
WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY corpus.py mcp_client.py rag_service.py mcp_server.py agent_harness_base.py index_docs.py skills.py ./

# No ENTRYPOINT on purpose: compose overrides the command per service.
CMD ["python", "mcp_server.py"]
