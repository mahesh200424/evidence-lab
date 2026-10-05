#!/bin/zsh
cd "${0:A:h}"
exec .venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false --server.maxUploadSize 20
