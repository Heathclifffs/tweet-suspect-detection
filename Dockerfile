FROM python:3.13-slim

RUN pip install uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY . .

EXPOSE 7860

CMD ["uv", "run", "streamlit", "run", "src/deploy/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
