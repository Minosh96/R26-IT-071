# Runs the app on a phone and points it at the backends on this PC.
#
# Usage (from the watinakama_app folder):
#   powershell -ExecutionPolicy Bypass -File run_on_phone.ps1        # phone on the same Wi-Fi as the PC
#   powershell -ExecutionPolicy Bypass -File run_on_phone.ps1 -Usb   # phone plugged in by USB: works on any Wi-Fi
# Extra flutter arguments are passed through, e.g. -d <device-id>

param([switch]$Usb)

$ports = 8000, 8080, 5003, 5004

if ($Usb) {
    # Forward the phone's localhost ports to this PC through the USB cable,
    # so the app can use 127.0.0.1 no matter which Wi-Fi either device is on.
    $adb = Join-Path $env:LOCALAPPDATA "Android\Sdk\platform-tools\adb.exe"
    if (-not (Test-Path $adb)) { $adb = (Get-Command adb -ErrorAction SilentlyContinue).Source }
    if (-not $adb) { Write-Host "adb not found. Install Android platform-tools." -ForegroundColor Red; exit 1 }
    foreach ($p in $ports) { & $adb reverse "tcp:$p" "tcp:$p" | Out-Null }
    Write-Host "USB mode: phone ports $($ports -join ', ') forwarded to this PC" -ForegroundColor Green
    flutter run --dart-define=API_HOST=127.0.0.1 @args
    exit $LASTEXITCODE
}

# The PC's Wi-Fi address changes when it joins a different network, so look it up every time.
$ip = Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias 'Wi-Fi*' -ErrorAction SilentlyContinue |
    Where-Object { $_.IPAddress -notlike '169.254.*' } |
    Select-Object -First 1 -ExpandProperty IPAddress

if (-not $ip) {
    Write-Host "Could not find this PC's Wi-Fi address. Is Wi-Fi connected? (Or use -Usb.)" -ForegroundColor Red
    exit 1
}

$network = (Get-NetConnectionProfile -InterfaceAlias 'Wi-Fi*' -ErrorAction SilentlyContinue | Select-Object -First 1).Name
Write-Host "PC Wi-Fi: $network  ->  backends at http://${ip}" -ForegroundColor Green
Write-Host "Make sure the phone is on the same Wi-Fi ($network)." -ForegroundColor Yellow

flutter run --dart-define=API_HOST=$ip @args
