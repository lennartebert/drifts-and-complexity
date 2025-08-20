# bootstrap.ps1 — Windows-only: conda hooks + optional CRAN install (iNEXT, ggplot2)
# Usage:
#   pwsh -File .\bootstrap.ps1              # just install hooks
#   pwsh -File .\bootstrap.ps1 -InstallCRAN # also install iNEXT & ggplot2 from CRAN (Windows binaries)

[CmdletBinding()]
param(
  [string]$EnvName = "drifts-and-complexity",
  [switch]$InstallCRAN
)

$ErrorActionPreference = "Stop"

function Resolve-CondaPrefix {
  try {
    $p = (conda run -n $EnvName python -c "import sys,os; print(os.environ.get('CONDA_PREFIX', sys.prefix))" 2>$null)
    if ($p) { return $p.Trim() }
  } catch {}
  try {
    $info = conda info --json | ConvertFrom-Json
    $candidate = @($info.envs) | Where-Object { $_ -like "*\envs\$EnvName" -or $_.Split('\')[-1] -eq $EnvName } | Select-Object -First 1
    if ($candidate) { return $candidate.Trim() }
  } catch {}
  return ""
}

# 0) ensure conda
$null = Get-Command conda -ErrorAction Stop

# 1) create/update env
if (conda env list | Select-String -Pattern "^\s*$EnvName\s" -Quiet) {
  Write-Host "Updating env $EnvName ..."
  conda env update -f environment.yml -n $EnvName --prune
} else {
  Write-Host "Creating env $EnvName ..."
  conda env create -f environment.yml
}

# 2) prefix
$prefix = [string](Resolve-CondaPrefix)
if ([string]::IsNullOrWhiteSpace($prefix)) { throw "Could not resolve prefix for env '$EnvName'." }
$prefix = ($prefix -replace '\u0000','').Trim()
Write-Host "Env prefix: [$prefix] (len=$($prefix.Length))"

# 3) hook dirs
$actDir   = "$prefix\etc\conda\activate.d"
$deactDir = "$prefix\etc\conda\deactivate.d"
New-Item -ItemType Directory -Force -Path $actDir,$deactDir | Out-Null

# 4) activation/deactivation hooks (PowerShell)
$activatePs1 = @'
# r-bridge.ps1 — runs on "conda activate"
$env:__OLD_PATH           = $env:PATH
$env:__OLD_R_HOME         = $env:R_HOME
$env:__OLD_RPY2_CFFI_MODE = $env:RPY2_CFFI_MODE
$env:__OLD_R_LIBS         = $env:R_LIBS
$env:__OLD_R_LIBS_USER    = $env:R_LIBS_USER

# Point to conda-forge R in this env
$env:R_HOME = Join-Path $env:CONDA_PREFIX "lib\R"

# Prepend R & MSYS2 bins (Rscript.exe, dlls, sh.exe)
$Rbin1 = Join-Path $env:CONDA_PREFIX "lib\R\bin"
$Rbin2 = Join-Path $env:CONDA_PREFIX "Library\bin"
$Rbin3 = Join-Path $env:CONDA_PREFIX "Library\usr\bin"
$env:PATH = "$Rbin1;$Rbin2;$Rbin3;$env:PATH"

# Force rpy2 to use ABI mode on Windows (prevents 'sh' lookup)
$env:RPY2_CFFI_MODE = "ABI"

# IMPORTANT: avoid cross-OS R libraries leaking in
Remove-Item Env:R_LIBS -ErrorAction SilentlyContinue
Remove-Item Env:R_LIBS_USER -ErrorAction SilentlyContinue
'@

$deactivatePs1 = @'
# r-bridge-deact.ps1 — runs on "conda deactivate"
if ($env:__OLD_PATH)           { $env:PATH = $env:__OLD_PATH;           Remove-Item Env:__OLD_PATH -ErrorAction SilentlyContinue }
if ($env:__OLD_R_HOME)         { $env:R_HOME = $env:__OLD_R_HOME }      else { Remove-Item Env:R_HOME -ErrorAction SilentlyContinue }
if ($env:__OLD_RPY2_CFFI_MODE) { $env:RPY2_CFFI_MODE = $env:__OLD_RPY2_CFFI_MODE } else { Remove-Item Env:RPY2_CFFI_MODE -ErrorAction SilentlyContinue }

# restore R_LIBS*
if ($env:__OLD_R_LIBS)      { $env:R_LIBS      = $env:__OLD_R_LIBS }      else { Remove-Item Env:R_LIBS      -ErrorAction SilentlyContinue }
if ($env:__OLD_R_LIBS_USER) { $env:R_LIBS_USER = $env:__OLD_R_LIBS_USER } else { Remove-Item Env:R_LIBS_USER -ErrorAction SilentlyContinue }
Remove-Item Env:__OLD_R_HOME, Env:__OLD_RPY2_CFFI_MODE, Env:__OLD_R_LIBS, Env:__OLD_R_LIBS_USER -ErrorAction SilentlyContinue
'@

Set-Content -Path "$actDir\r-bridge.ps1"         -Value $activatePs1   -Encoding UTF8
Set-Content -Path "$deactDir\r-bridge-deact.ps1" -Value $deactivatePs1 -Encoding UTF8

# 5) ensure ABI at Python startup
$sitePkg = Join-Path $prefix "Lib\site-packages"
$siteCustomize = Join-Path $sitePkg "sitecustomize.py"
$siteContent = @'
import os
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
'@
New-Item -ItemType Directory -Force -Path $sitePkg | Out-Null
Set-Content -Path $siteCustomize -Value $siteContent -Encoding UTF8
Write-Host "Injected ABI-mode bootstrapper into $siteCustomize"

# 6) (optional) install clean Windows CRAN binaries of iNEXT + ggplot2 into **env library**
if ($InstallCRAN) {
  $tmpR = Join-Path $env:TEMP "install_inext.R"

  # R code (no BOM). Installs into %R_HOME%/library (the conda env's R lib)
  $rCode = @'
repos <- "https://cloud.r-project.org"
pkgs  <- c("ggplot2","iNEXT")

# Env library (inside this conda env)
lib <- file.path(Sys.getenv("R_HOME"), "library")
dir.create(lib, recursive = TRUE, showWarnings = FALSE)

# Make ENV lib first on the search path
.libPaths(c(lib, .libPaths()))

# Purge any existing iNEXT from other libraries to avoid arch mismatches
suppressWarnings({
  for (libdir in .libPaths()) {
    ip <- installed.packages(lib.loc = libdir)
    if (!is.null(ip) && "iNEXT" %in% rownames(ip)) {
      try(remove.packages("iNEXT", lib = libdir), silent = TRUE)
    }
  }
})

# Install Windows binaries into ENV lib
need <- setdiff(pkgs, rownames(installed.packages(lib.loc = .libPaths())))
if (length(need)) {
  install.packages(need, repos = repos, type = "binary", lib = lib, dependencies = TRUE)
}

cat("libPaths:", paste(.libPaths(), collapse=" | "), "\n")
for (p in pkgs) {
  if (requireNamespace(p, quietly = TRUE)) {
    cat(p, "->", as.character(utils::packageVersion(p)), "\n")
  } else {
    cat("Package not installed:", p, "\n")
  }
}
'@

  # Write the R script WITHOUT BOM
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [System.IO.File]::WriteAllText($tmpR, $rCode, $utf8NoBom)

  Write-Host "Installing ggplot2 + iNEXT (CRAN binaries) into ENV library..."
  conda run -n $EnvName Rscript --vanilla "$tmpR"
}


Write-Host "Hooks installed:"
Write-Host "  $actDir\r-bridge.ps1"
Write-Host "  $deactDir\r-bridge-deact.ps1"
Write-Host ""
Write-Host "Now refresh env so hooks take effect:"
Write-Host "  conda deactivate; conda activate $EnvName"
Write-Host ""
Write-Host "Quick check afterwards:"
Write-Host '  echo "R_HOME=$env:R_HOME"; echo "RPY2_CFFI_MODE=$env:RPY2_CFFI_MODE"; python -c "from rpy2.robjects.packages import importr; print(\"rpy2 OK\"); importr(\"iNEXT\"); print(\"iNEXT OK\")"'
