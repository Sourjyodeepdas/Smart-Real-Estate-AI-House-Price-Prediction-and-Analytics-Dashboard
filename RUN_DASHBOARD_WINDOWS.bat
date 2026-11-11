@echo off
cd /d "%~dp0"
python -m streamlit run app.py
pause
git init
git branch -M main
git add .
git commit -m "Initial commit: Smart Real Estate AI"