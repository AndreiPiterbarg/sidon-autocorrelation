Get-WmiObject Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -like '*_v2_v3_v4_bench*' } |
  ForEach-Object {
    Write-Host "Killing PID $($_.ProcessId): $($_.CommandLine)"
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
  }
