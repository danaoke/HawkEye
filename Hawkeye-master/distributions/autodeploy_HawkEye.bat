call C:\Programs\Anaconda3\Scripts\activate.bat 
call conda activate Hawkeye
call pip install -r Hawkeye_PackageRequirements.txt
set MYPATH=%CD%
cd ..\
call pyinstaller -y --clean --distpath "%MYPATH%" --workpath "%MYPATH%\build" HawkeyeGUI.spec

cd "%MYPATH%"
set /p confirmed="Compress Hawkeye for distribution? [y/n]"
IF %confirmed%==y (
	Xcopy /S /Y "..\Excel Templates\*" "Excel Templates\*"
	rmdir /Q /S "Excel Templates\oldversion inputs"
	powershell Compress-Archive -Path "HawkeyeGUI",'Excel Templates',"setup.bat" ^
	-DestinationPath "Hawkeye.zip" -Force
	rmdir /Q /S "Excel Templates"
	)

PAUSE