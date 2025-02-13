$action = New-ScheduledTaskAction `
    -Execute "C:\Users\69185\AppData\Local\Programs\Python\Python313\python.exe" `
    -Argument "c:\Users\69185\source\repos\sonet\src\backtest.py" `
    -WorkingDirectory "c:\Users\69185\source\repos\sonet\src"

$trigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Hours 4)

$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask `
    -TaskName "CryptoBacktest" `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Force