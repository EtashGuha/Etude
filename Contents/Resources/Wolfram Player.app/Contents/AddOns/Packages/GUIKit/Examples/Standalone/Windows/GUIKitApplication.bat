@echo off

REM   Copyright (c) 2003 Wolfram Research, Inc.  All rights
REM   reserved.
REM   Sample start script for a GUIKit standalone Application
REM
REM   Make sure MATH_HOME and GUIKIT_HOME below
REM   and optionally JAVA_HOME are defined correctly
REM   for your configuration
REM
REM   NOTE: you must pass GUIKit` definition relative names to this script 
REM    for it to actually run anything.  See the cooresponding PrimeFinderApplication.bat
REM    script for an example of a specific user interface definition launch

if "%OS%"=="Windows_NT" @setlocal

REM   Change this path to where Mathematica is located or define this in your environment
if "%MATH_HOME%" == "" set MATH_HOME=C:\Program Files\Wolfram Research\Mathematica\5.0

echo Using Mathematica from %MATH_HOME%

REM   Change this path to where you have installed the GUIKit` AddOn or define this in your environment
if "%GUIKIT_HOME%" == "" set GUIKIT_HOME=%MATH_HOME%\AddOns\Applications\GUIKit

echo Using GUIKit from %GUIKIT_HOME%

REM   If no JAVA_HOME it will default to the VM installed with Mathematica
if "%JAVA_HOME%" == "" set JAVA_HOME=%MATH_HOME%\SystemFiles\Java\Windows
set _JAVACMD=%JAVA_HOME%\bin\javaw.exe
set JAVA_OPTS=-showversion

echo Using Java VM from %JAVA_HOME%

set LOCALCLASSPATH=
if exist "%MATH_HOME%\AddOns\JLink\JLink.jar" set LOCALCLASSPATH=%MATH_HOME%\AddOns\JLink\JLink.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\bsf.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\bsf.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\bsf-Wolfram.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\bsf-Wolfram.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\diva-canvas-core.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\diva-canvas-core.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\concurrent.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\concurrent.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\GUIKit.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\GUIKit.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\OculusLayout.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\OculusLayout.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\xercesImpl.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\xercesImpl.jar;%LOCALCLASSPATH%
if exist "%GUIKIT_HOME%\Java\xmlParserAPIs.jar" set LOCALCLASSPATH=%GUIKIT_HOME%\Java\xmlParserAPIs.jar;%LOCALCLASSPATH%

"%_JAVACMD%" %JAVA_OPTS% -cp "%LOCALCLASSPATH%" "-Dbsf_engine_Mathematica_KernelLinkCommandLine=-linkmode launch -linkname '%MATH_HOME%\MathKernel.exe'" com.wolfram.guikit.app.GUIKitApplication %1 %2 %3 %4 %5 %6 %7 %8 %9

if "%OS%"=="Windows_NT" @endlocal
