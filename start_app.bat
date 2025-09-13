@echo off
REM CAVEO Chatbot Launcher
REM Dark mode enforced via .streamlit/config.toml

echo ====================================
echo  CAVEO Chatbot
echo ====================================
echo.
echo Starting CAVEO Chatbot (local, offline)...
echo.

REM Set environment variables to reduce warnings
set TORCH_LOGS=0
set PYTHONWARNINGS=ignore::UserWarning:torch,ignore::DeprecationWarning:langchain

REM Start the application
cd /d "%~dp0"
streamlit run app.py --server.headless=false

echo.
echo Application stopped.
pause
