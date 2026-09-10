@echo off
REM The control API must start from the repo root: launched from D:\ the
REM name "clipforge" resolves to the DIRECTORY as a namespace package and
REM every import fails. The test suite has the same requirement.
cd /d "%~dp0"
"D:\clipforge\.venv\Scripts\python.exe" -m clipforge.cli web %*
