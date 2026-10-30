param(
    [string]$Project = 'project-fd312a16-8445-4b1f-862',
    [string]$Region = 'asia-south1',
    [string]$Model = 'nvidia/Nemotron-3_5-Lightning',
    [ValidateRange(1, 600)]
    [int]$NebiusTimeoutSeconds = 240,
    [ValidateRange(1, 1800)]
    [int]$NebiusGenerationDeadlineSeconds = 600,
    [string]$Service = 'prism-spectra-ai',
    [string]$NebiusSecret = 'nebius-api-key',
    [string]$TavilySecret = '',
    [string]$LangSmithSecret = '',
    [string]$LangSmithWorkspaceSecret = '',
    [ValidateSet('https://api.smith.langchain.com', 'https://eu.api.smith.langchain.com')]
    [string]$LangSmithEndpoint = 'https://api.smith.langchain.com',
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
$Tracing = if ($LangSmithSecret) { 'true' } else { 'false' }
$Variables = "AI_PROVIDER=nebius,NEBIUS_MODEL=$Model,NEBIUS_TIMEOUT_SECONDS=$NebiusTimeoutSeconds,NEBIUS_GENERATION_DEADLINE_SECONDS=$NebiusGenerationDeadlineSeconds,EMBEDDING_PROVIDER=nebius,NEBIUS_EMBED_MODEL=Qwen/Qwen3-Embedding-8B,USE_FIRESTORE=1,FIRESTORE_DATABASE=$FirestoreDatabase,SEED_DEMO=1,USE_SECRET_MANAGER=0,GOOGLE_CLOUD_PROJECT=$Project,GOOGLE_CLOUD_LOCATION=global,GCS_BUCKET=$($Config.bucket),BQ_DATASET=$($Config.dataset),KAIROS_ADMIN_EMAIL=$AdminEmail,FORCE_MOCK=0,LANGSMITH_TRACING=$Tracing,LANGSMITH_PROJECT=prism-spectra-ai,LANGSMITH_ENDPOINT=$LangSmithEndpoint,LANGSMITH_HIDE_INPUTS=true,LANGSMITH_HIDE_OUTPUTS=true"
$Secrets = "NEBIUS_API_KEY=$($NebiusSecret):latest,KAIROS_ADMIN_PASSWORD=$($AdminPasswordSecret):latest"
if ($TavilySecret) {
    $Secrets += ",TAVILY_API_KEY=$($TavilySecret):latest"
}
if ($LangSmithSecret) {
    $Secrets += ",LANGSMITH_API_KEY=$($LangSmithSecret):latest"
}
if ($LangSmithWorkspaceSecret) {
    $Secrets += ",LANGSMITH_WORKSPACE_ID=$($LangSmithWorkspaceSecret):latest"
}
& $Python $Cli run deploy $Service --source . --project=$Project --region=$Region `
    --build-service-account="projects/$Project/serviceAccounts/$($Config.build_account)" `
    --service-account=$($Config.service_account) --allow-unauthenticated `
    --set-env-vars=$Variables --set-secrets=$Secrets --memory=1Gi --cpu=1 --min-instances=0 `
    --max-instances=$MaxInstances --concurrency=8 --session-affinity --timeout=3600 --port=8080 --quiet
if ($LASTEXITCODE -ne 0) { throw 'Cloud Run deployment failed.' }
& $Python $Cli run services describe $Service --project=$Project --region=$Region --format='value(status.url)'
if ($LASTEXITCODE -ne 0) { throw 'Could not retrieve Cloud Run URL.' }
