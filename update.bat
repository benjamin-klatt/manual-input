@echo off
REM Activate venv
call .\.venv\Scripts\Activate.ps1

REM Stash configs if modified
git diff --quiet configs || git stash push -m "local configs" configs

REM Pull latest, force overwrite everything except configs
git fetch
git checkout origin/main -- .
git reset --hard origin/main

REM Restore configs if stashed
git stash list | findstr "local configs" >nul
if %errorlevel%==0 git stash pop

REM Install requirements
.\.venv\Scripts\pip install -r requirements.txt