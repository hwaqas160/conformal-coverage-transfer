# Usage: powershell -File run\schedule.ps1 -Name P01_TrainCpuV2 -Cmd F:\...\run\train_cpu_v2.cmd [-DelayMin 1]
# Registers a one-shot Task Scheduler job (own session => survives the CLI/host ending) with no time
# limit and no battery restrictions. Re-registering an existing name replaces it (-F).
param([Parameter(Mandatory=$true)][string]$Name,[Parameter(Mandatory=$true)][string]$Cmd,[int]$DelayMin=1)
$start = (Get-Date).AddMinutes($DelayMin).ToString('HH:mm')
schtasks /Create /TN $Name /TR $Cmd /SC ONCE /ST $start /F /RL LIMITED | Out-Null
$t = Get-ScheduledTask -TaskName $Name
$t.Settings.DisallowStartIfOnBatteries = $false
$t.Settings.StopIfGoingOnBatteries = $false
$t.Settings.ExecutionTimeLimit = "PT0S"
$t.Settings.MultipleInstances = "IgnoreNew"
Set-ScheduledTask -TaskName $Name -Settings $t.Settings | Out-Null
Write-Output "scheduled $Name at $start -> $Cmd"
