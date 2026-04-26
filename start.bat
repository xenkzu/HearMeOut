@echo off
echo ======================================================
echo          HEARMEOUT - ANALYZER TOOL
echo ======================================================
echo.
set /p filename="> Enter filename (from HearMeOutTesting folder): "
echo.
echo Processing %filename%...
python src-python\process_file.py "%filename%"
echo.
echo Press any key to exit.
pause > nul
