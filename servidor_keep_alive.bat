@echo off
title Watchdog do Reconhecimento Facial
color 0A

:: Nome do executável empacotado
set APP_PATH=dist\app\app.exe

echo ====================================================
echo Inicializador do Servidor - Modo de Reinicializacao
echo ====================================================
echo.

:loop
echo [%time%] Iniciando a aplicacao Face.ID...
:: "start /wait" garante que o script pause e so continue quando o app for fechado
start /wait "" "%APP_PATH%"

echo ====================================================
echo [%time%] ATENCAO: Aplicacao fechou ou foi derrubada!
echo Reiniciando o sistema automaticamente em 5 segundos...
echo ====================================================
timeout /t 5 /nobreak > nul

goto loop
