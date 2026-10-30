param(
    [string]$Project = 'project-fd312a16-8445-4b1f-862',
    [string]$Region = 'asia-south1',
    [string]$Model = 'nvidia/Nemotron-3_5-Lightning',
    [string]$Service = 'prism-spectra-ai',
    [string]$NebiusSecret = 'nebius-api-key',
    [string]$TavilySecret = '',
    [string]$FirestoreDatabase = 'prism-spectra-demo',
    [string]$AdminEmail = 'admin@kairos.ai',
    [Parameter(Mandatory = $true)]
    [string]$AdminPasswordSecret,
    [ValidateRange(1, 4)]
    [int]$MaxInstances = 2
)
$ErrorActionPreference = 'Stop'
$PythonCommand = Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1
$Python = $PythonCommand.Source
$Cli = Join-Path $env:LOCALAPPDATA 'Google\Cloud SDK\google-cloud-sdk\lib\gcloud.py'
$CaBundle = Join-Path $env:APPDATA 'gcloud\windows-trusted-roots.pem'
if (Test-Path -LiteralPath $CaBundle) {
    $env:REQUESTS_CA_BUNDLE = $CaBundle
}
Set-Location (Split-Path $PSScriptRoot -Parent)
$Config = Get-Content (Join-Path $PSScriptRoot 'deployment.json') -Raw | ConvertFrom-Json
if ($FirestoreDatabase -eq '(default)') {
    throw 'Use a separate named Firestore database for the public demo, not existing private device storage.'
}
$Variables = "AI_PROVIDER=nebius,NEBIUS_MODEL=$Model,EMBEDDING_PROVIDER=nebius,NEBIUS_EMBED_MODEL=Qwen/Qwen3-Embedding-8B,USE_VERTEX=0,USE_FIRESTORE=1,FIRESTORE_DATABASE=$FirestoreDatabase,SEED_DEMO=1,USE_SECRET_MANAGER=0,GOOGLE_CLOUD_PROJECT=$Project,GOOGLE_CLOUD_LOCATION=global,GCS_BUCKET=$($Config.bucket),BQ_DATASET=$($Config.dataset),KAIROS_ADMIN_EMAIL=$AdminEmail,FORCE_MOCK=0"
$Secrets = "NEBIUS_API_KEY=$($NebiusSecret):latest,KAIROS_ADMIN_PASSWORD=$($AdminPasswordSecret):latest"
if ($TavilySecret) {
    $Secrets += ",TAVILY_API_KEY=$($TavilySecret):latest"
}
& $Python $Cli run deploy $Service --source . --project=$Project --region=$Region `
    --build-service-account="projects/$Project/serviceAccounts/$($Config.build_account)" `
    --service-account=$($Config.service_account) --allow-unauthenticated `
    --set-env-vars=$Variables --set-secrets=$Secrets --memory=1Gi --cpu=1 --min-instances=0 `
    --max-instances=$MaxInstances --concurrency=8 --timeout=3600 --port=8080 --quiet
if ($LASTEXITCODE -ne 0) { throw 'Cloud Run deployment failed.' }
& $Python $Cli run services describe $Service --project=$Project --region=$Region --format='value(status.url)'
if ($LASTEXITCODE -ne 0) { throw 'Could not retrieve Cloud Run URL.' }
