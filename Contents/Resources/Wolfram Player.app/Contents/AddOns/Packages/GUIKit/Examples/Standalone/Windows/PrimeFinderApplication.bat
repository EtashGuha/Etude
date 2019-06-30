@echo off

REM   Copyright (c) 2003 Wolfram Research, Inc.  All rights
REM   reserved.
REM   
REM  This is simply a demo of reusing the adjacent GUIKitApplication.bat script
REM  to launch a specific user interface definition

if "%OS%"=="Windows_NT" @setlocal

.\GUIKitApplication.bat "Wolfram/Example/PrimeFinder"

if "%OS%"=="Windows_NT" @endlocal
