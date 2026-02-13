@echo off
REM Stop the distributed system

echo Stopping Heart Attack Anomaly Detection System...
docker-compose down

echo.
echo System stopped successfully!
pause
