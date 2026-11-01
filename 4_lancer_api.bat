@echo off
echo Documentation : http://127.0.0.1:8000/docs
python -m uvicorn api.main:app --reload
