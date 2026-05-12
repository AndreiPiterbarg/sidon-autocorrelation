Get-WmiObject Win32_Process -Filter "Name = 'python.exe'" |
  Select-Object ProcessId, CommandLine, WorkingSetSize |
  Format-Table -AutoSize -Wrap |
  Out-String -Width 220
