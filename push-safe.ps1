<#
.SYNOPSIS
  Safe one-file Git push script for Windows PowerShell.

.DESCRIPTION
  Shows changed files, asks for confirmation, commits and pushes to the current branch.
  Usage:
    .\push-safe.ps1            -> interactive (recommended)
    .\push-safe.ps1 -Message "my commit"  -> interactive, prefilled message
    .\push-safe.ps1 -Auto -Message "msg"  -> non-interactive: add, commit, push

.NOTES
  Author: Mohd Arshad helper script
#>

param(
    [string]$Message = "",
    [switch]$Auto
)

function Write-Info($s) { Write-Host $s -ForegroundColor Cyan }
function Write-Warn($s) { Write-Host $s -ForegroundColor Yellow }
function Write-Err($s)  { Write-Host $s -ForegroundColor Red }

# Ensure we're in a git repo
if (-not (Test-Path ".git")) {
    Write-Err "This folder does not appear to be a git repository (no .git directory). Run this script from the repository root."
    exit 2
}

# Get current branch
$branch = git rev-parse --abbrev-ref HEAD 2>$null
if ($LASTEXITCODE -ne 0 -or -not $branch) {
    Write-Err "Failed to determine current branch."
    exit 3
}
Write-Info "Current branch: $branch"

# Get git status porcelain
$status = git status --porcelain
if (-not $status) {
    Write-Warn "No changes to commit. Nothing to push."
    exit 0
}

Write-Info "The following changes are detected (git status --porcelain):"
$statusLines = $status -split "`n"
foreach ($line in $statusLines) {
    if ($line.Trim()) { Write-Host "  $line" }
}

if (-not $Auto) {
    $confirm = Read-Host "Proceed to add, commit and push these changes to '$branch'? (Y/N)"
    if ($confirm.Trim().ToUpper() -ne "Y") {
        Write-Warn "Aborted by user."
        exit 0
    }
}

# Prepare commit message
if (-not $Message) {
    if ($Auto) {
        $Message = "Auto commit on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    } else {
        $msgInput = Read-Host "Enter commit message (leave empty to use timestamp)"
        if ($msgInput.Trim()) { $Message = $msgInput } else { $Message = "Commit on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" }
    }
}

Write-Info "Using commit message: $Message"

# Stage changes
Write-Info "Staging changes..."
git add -A
if ($LASTEXITCODE -ne 0) {
    Write-Err "git add failed."
    exit 4
}

# Commit (if nothing new to commit, warn and continue to push)
git commit -m "$Message" 2>$null
if ($LASTEXITCODE -ne 0) {
    # check if commit failed because there was nothing to commit
    $diffAfterAdd = git diff --cached --name-only
    if (-not $diffAfterAdd) {
        Write-Warn "No staged changes to commit (maybe previous commit already included them). Proceeding to push."
    } else {
        Write-Err "git commit failed."
        exit 5
    }
} else {
    Write-Info "Commit created."
}

# Push
Write-Info "Pushing to origin/$branch ..."
git push origin $branch
if ($LASTEXITCODE -ne 0) {
    Write-Err "git push failed. Check your network, credentials, or remote settings."
    exit 6
}

Write-Host ""
Write-Host "✅ Successfully pushed to origin/$branch" -ForegroundColor Green
exit 0
