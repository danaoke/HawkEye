@echo off
set MYPATH=%CD%

echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%mypath%\Hawkeye.lnk" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "%mypath%\HawkeyeGUI\HawkeyeGUI.exe" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%mypath%" >> CreateShortcut.vbs
echo oLink.IconLocation = "%mypath%\HawkeyeGUI\hawkeye.ico" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs
cscript CreateShortcut.vbs
del CreateShortcut.vbs

echo Created Hawkeye shortcut to %mypath%
set /p confirmed="Create shortcut to Desktop? [y/n]"
IF %confirmed%==y (
	copy Hawkeye.lnk %USERPROFILE%\Desktop\Hawkeye.lnk"
	echo Created Hawkeye shortcut to %USERPROFILE%\Desktop
	)
PAUSE
