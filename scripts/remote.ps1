[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('doctor','deploy','setup','smoke','pilot','fixtures','status','fetch')]
    [string]$Action,
    [string]$ConfigFile = '',
    [string]$RunId = '',
    [switch]$Wait,
    [int]$WaitTimeoutSeconds = 7500
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$Repo = Split-Path -Parent $PSScriptRoot
if (-not $ConfigFile) { $ConfigFile = Join-Path $Repo 'configs/machine.local.json' }
if (-not (Test-Path -LiteralPath $ConfigFile)) {
    throw "Copy configs/machine.example.json to configs/machine.local.json and fill in this target. Missing: $ConfigFile"
}
$Machine = Get-Content -LiteralPath $ConfigFile -Raw | ConvertFrom-Json
foreach ($field in @('host','port','user','identity_file','expected_hostname','remote_root','scratch_root')) {
    if (-not $Machine.PSObject.Properties[$field] -or -not $Machine.$field) { throw "Missing machine config field: $field" }
}
foreach ($field in @('host','user','expected_hostname')) {
    if ($Machine.$field -notmatch '^[A-Za-z0-9_.@-]+$' -or $Machine.$field.StartsWith('-')) { throw "Invalid $field" }
}
foreach ($field in @('remote_root','scratch_root')) {
    if ($Machine.$field -notmatch '^/[A-Za-z0-9_./-]+$' -or $Machine.$field -match '(^|/)\.\.(/|$)') { throw "Use a simple absolute POSIX $field without spaces or traversal" }
    $Machine.$field = $Machine.$field.TrimEnd('/')
}
if ($Machine.scratch_root -notmatch '^/tmp/.+') { throw 'scratch_root must be a task-owned directory under /tmp' }
if ([int]$Machine.port -lt 1 -or [int]$Machine.port -gt 65535) { throw 'Invalid SSH port' }
if (-not (Test-Path -LiteralPath $Machine.identity_file -PathType Leaf)) { throw "SSH identity is not readable at the configured local path" }
$Target = "$($Machine.user)@$($Machine.host)"
$SshArguments = @('-o','BatchMode=yes','-o','ConnectTimeout=20','-o','IdentitiesOnly=yes','-i',[string]$Machine.identity_file,'-p',[string]$Machine.port,$Target)
$ScpArguments = @('-q','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','IdentitiesOnly=yes','-i',[string]$Machine.identity_file,'-P',[string]$Machine.port)
# EFabric's Paramiko gateway requires legacy SCP; ordinary OpenSSH targets can select sftp.
$ScpProtocol = 'legacy'
if ($Machine.PSObject.Properties['scp_protocol']) { $ScpProtocol = [string]$Machine.scp_protocol }
if ($ScpProtocol -notin @('legacy','sftp')) { throw 'scp_protocol must be legacy or sftp' }
if ($ScpProtocol -eq 'legacy') { $ScpArguments = @('-O') + $ScpArguments }
# EFabric's Paramiko gateway requires legacy SCP; ordinary OpenSSH targets can select sftp.
$ScpProtocol = 'legacy'
if ($Machine.PSObject.Properties['scp_protocol']) { $ScpProtocol = [string]$Machine.scp_protocol }
if ($ScpProtocol -notin @('legacy','sftp')) { throw 'scp_protocol must be legacy or sftp' }
if ($ScpProtocol -eq 'legacy') { $ScpArguments = @('-O') + $ScpArguments }
$ControllerDir = Join-Path $Repo '.controller'
New-Item -ItemType Directory -Force -Path $ControllerDir | Out-Null
$StatePath = Join-Path $ControllerDir 'state.json'
$State = @{}
if (Test-Path -LiteralPath $StatePath) {
    $old = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
    foreach ($property in $old.PSObject.Properties) { $State[$property.Name] = $property.Value }
}
$TargetIdentity = "$Target`:$($Machine.port)$($Machine.remote_root)"
if ($State.ContainsKey('target') -and $State.target -ne $TargetIdentity) { $State = @{} }
$State.target = $TargetIdentity

function Save-State {
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($StatePath, ($State | ConvertTo-Json -Depth 12), $utf8)
}
function Quote-Remote([string]$Text) {
    if ($Text -match "['`r`n]") { throw 'Remote argument contains forbidden quote/newline' }
    return "'$Text'"
}
function Invoke-Ssh([string[]]$RemoteArguments) {
    $command = ($RemoteArguments | ForEach-Object { Quote-Remote $_ }) -join ' '
    $lines = & ssh @SshArguments $command
    if ($LASTEXITCODE -ne 0) { throw "SSH command failed with exit $LASTEXITCODE; no experiment success is implied" }
    return ($lines -join "`n")
}
function Invoke-Scp([string]$From, [string]$To) {
    & scp @ScpArguments $From $To
    if ($LASTEXITCODE -ne 0) { throw "SCP transfer failed with exit $LASTEXITCODE" }
}
function Invoke-LocalPython([string[]]$PythonArguments) {
    $lines = & python @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Local Python command failed with exit $LASTEXITCODE" }
    return ($lines -join "`n")
}
function Invoke-Control([string[]]$ControlArguments) {
    if (-not $State.ContainsKey('release_id')) { throw 'No deployed release recorded; run deploy first' }
    $script = "$($Machine.remote_root)/releases/$($State.release_id)/scripts/control.py"
    $arguments = @('python3',$script,'--root',[string]$Machine.remote_root,'--expected-hostname',[string]$Machine.expected_hostname) + $ControlArguments
    return ((Invoke-Ssh $arguments) | ConvertFrom-Json)
}
function Deploy-Snapshot {
    $snapshot = (Invoke-LocalPython @((Join-Path $PSScriptRoot 'snapshot.py'),'create','--root',$Repo,'--output',(Join-Path $ControllerDir 'snapshots'))) | ConvertFrom-Json
    $incoming = "$($Machine.remote_root)/incoming"
    $boot = "$incoming/bootstrap-$($snapshot.release_id)"
    Invoke-Ssh @('mkdir','-p',$boot) | Out-Null
    Invoke-Scp (Join-Path $PSScriptRoot 'control.py') "${Target}:$boot/control.py"
    Invoke-Scp (Join-Path $PSScriptRoot 'snapshot.py') "${Target}:$boot/snapshot.py"
    Invoke-Scp $snapshot.archive "${Target}:$incoming/$($snapshot.release_id).tar.gz"
    $receipt = (Invoke-Ssh @('python3',"$boot/control.py",'--root',[string]$Machine.remote_root,'--expected-hostname',[string]$Machine.expected_hostname,'deploy','--release',[string]$snapshot.release_id,'--sha256',[string]$snapshot.archive_sha256)) | ConvertFrom-Json
    if ($receipt.archive_sha256 -ne $snapshot.archive_sha256 -or $receipt.file_count -ne $snapshot.file_count) { throw 'Remote deployment receipt mismatch' }
    $State.release_id = $snapshot.release_id
    $State.archive_sha256 = $snapshot.archive_sha256
    Save-State
    return $receipt
}

$actualHostname = (Invoke-Ssh @('hostname')).Trim()
if ($actualHostname -ne $Machine.expected_hostname) { throw "SSH hostname mismatch: expected $($Machine.expected_hostname), got $actualHostname" }

if ($Action -eq 'deploy') {
    Deploy-Snapshot | ConvertTo-Json -Depth 12
    exit 0
}
if ($Action -in @('doctor','setup','smoke','pilot','fixtures')) {
    $deployment = Deploy-Snapshot
    Write-Host "Verified release $($deployment.release_id) ($($deployment.file_count) source files)."
    # Record the intended ID before SSH so a lost launch reply can be reconciled after reconnecting.
    $pendingRun = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ') + "-$Action-" + [Guid]::NewGuid().ToString('N').Substring(0,8)
    $State.latest_run_id = $pendingRun
    $State["${Action}_run_id"] = $pendingRun
    Save-State
    $launch = Invoke-Control @('launch','--release',[string]$State.release_id,'--kind',$Action,'--scratch',[string]$Machine.scratch_root,'--run',$pendingRun)
    $launch | ConvertTo-Json -Depth 12
    if (-not $Wait) { exit 0 }
    $RunId = $launch.run_id
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    $nextUpdate = 0
    do {
        $status = Invoke-Control @('status','--run',$RunId)
        if ($status.state -in @('completed','failed','lost')) { break }
        if ($timer.Elapsed.TotalSeconds -ge $nextUpdate) {
            Write-Host "Run $RunId is $($status.state); elapsed $([int]$timer.Elapsed.TotalSeconds)s. Logs survive reconnects."
            $nextUpdate += 30
        }
        if ($timer.Elapsed.TotalSeconds -gt $WaitTimeoutSeconds) { throw "Local waiting deadline reached. Remote run may still be running; use status -RunId $RunId" }
        Start-Sleep -Seconds 5
    } while ($true)
    $status | ConvertTo-Json -Depth 12
    if ($status.state -ne 'completed' -or $status.exit_code -ne 0) { throw "Remote run did not succeed: $($status.state), exit $($status.exit_code). Fetch its logs." }
    exit 0
}
if (-not $RunId) {
    if (-not $State.ContainsKey('latest_run_id')) { throw 'Specify -RunId; no previous run is recorded for this target' }
    $RunId = [string]$State.latest_run_id
}
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$') { throw 'Invalid run ID' }
if ($Action -eq 'status') {
    Invoke-Control @('status','--run',$RunId) | ConvertTo-Json -Depth 12
    exit 0
}
if ($Action -eq 'fetch') {
    $packed = Invoke-Control @('pack','--run',$RunId)
    $download = Join-Path $ControllerDir "$RunId-results.tar.gz"
    Invoke-Scp "${Target}:$($packed.archive)" $download
    $destination = Join-Path (Join-Path $Repo 'results') $RunId
    $receipt = Invoke-LocalPython @((Join-Path $PSScriptRoot 'snapshot.py'),'extract','--archive',$download,'--sha256',[string]$packed.archive_sha256,'--output',$destination)
    Write-Host "Verified compact results: $destination"
    $receipt
}
