# Usage: powershell -File run\install_watchdog.ps1 -Name P01_TrainAv2GpuFull -Cmd F:\...\run\train_av2_gpu_full.cmd [-EveryMin 10]
# Registers a Task Scheduler task that (re)starts -Cmd now and then EVERY -EveryMin minutes for 60 days.
# MultipleInstances=IgnoreNew: a tick while the job is alive is a no-op; a tick after the job's process tree was
# killed (logoff / shutdown attempt / crash) starts it again.  The command itself must therefore be idempotent and
# must end the cycle with a DONE/FAILED marker (see run\train_av2_gpu_full.cmd).  No time limit, runs on battery.
param([Parameter(Mandatory=$true)][string]$Name,[Parameter(Mandatory=$true)][string]$Cmd,[int]$EveryMin=10)
$action  = New-ScheduledTaskAction -Execute $Cmd
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) -RepetitionDuration (New-TimeSpan -Days 60)
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Seconds 0) `
            -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Force | Out-Null
$t = Get-ScheduledTask -TaskName $Name
Write-Output ("watchdog {0}: state={1}, every {2} min, first run {3:HH:mm}, cmd={4}" -f $Name, $t.State, $EveryMin, $trigger.StartBoundary, $Cmd)
