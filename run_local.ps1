$ErrorActionPreference = 'Stop'
$Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

foreach ($Name in @('REQUESTS_CA_BUNDLE', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE')) {
    $CaBundle = [Environment]::GetEnvironmentVariable($Name, 'Process')
    if ($CaBundle -and -not (Test-Path -LiteralPath $CaBundle -PathType Leaf)) {
        Write-Warning "$Name points to a missing CA bundle; using the system certificate store instead."
        [Environment]::SetEnvironmentVariable($Name, $null, 'Process')
    }
}

if (-not (Test-Path $Python)) {
    $BootstrapPython = Get-Command python -CommandType Application -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $BootstrapPython) {
        throw 'Python was not found on PATH. Install Python 3.11 or newer, then run this launcher again.'
    }

    Write-Output 'Creating the project virtual environment...'
    & $BootstrapPython.Source -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $Python)) {
        throw 'Could not create .venv. Check that the Python installation includes the venv module.'
    }
}

$InstallMarker = Join-Path $PSScriptRoot '.venv\requirements-installed'
if (-not (Test-Path $InstallMarker)) {
    Write-Output 'Installing project dependencies (first launch only)...'
    & $Python -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependency installation failed. Check your network connection and rerun this launcher.'
    }
    New-Item -ItemType File -Path $InstallMarker -Force | Out-Null
}

$env:AI_PROVIDER = 'nebius'
$env:EMBEDDING_PROVIDER = 'nebius'
$env:FORCE_MOCK = '0'
Set-Location $PSScriptRoot

Write-Output 'Starting the Nebius-backed Prism-Spectra-AI demo on http://localhost:8506.'
Write-Output 'The Token Factory key must be configured locally in .env or NEBIUS_API_KEY.'
& $Python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8506 --server.headless=true --browser.gatherUsageStats=false
exit $LASTEXITCODE
