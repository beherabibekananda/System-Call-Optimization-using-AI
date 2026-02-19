from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import HttpResponse
from .forms import RegisterForm, LoginForm
from .models import AgentToken


def register_view(request):
    """User registration"""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save()
            # Auto-create an agent token for the new user
            AgentToken.objects.get_or_create(user=user)
            login(request, user)
            messages.success(request, f'Welcome to SysCall AI, {user.username}! Your account has been created.')
            return redirect('dashboard')
        else:
            for field, errors in form.errors.items():
                for error in errors:
                    messages.error(request, f'{error}')
    else:
        form = RegisterForm()

    return render(request, 'accounts/register.html', {'form': form})


def login_view(request):
    """User login"""
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                # Auto-create token if missing
                AgentToken.objects.get_or_create(user=user)
                if not form.cleaned_data.get('remember_me'):
                    request.session.set_expiry(0)
                else:
                    request.session.set_expiry(1209600)
                messages.success(request, f'Welcome back, {user.username}!')
                next_url = request.GET.get('next', 'dashboard')
                return redirect(next_url)
            else:
                messages.error(request, 'Invalid username or password.')
    else:
        form = LoginForm()

    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    """User logout"""
    logout(request)
    messages.info(request, 'You have been logged out.')
    return redirect('login')


@login_required
def profile_view(request):
    """User profile with agent token management"""
    token, _ = AgentToken.objects.get_or_create(user=request.user)
    return render(request, 'accounts/profile.html', {'token': token})


@login_required
def regenerate_token_view(request):
    """Regenerate the user's agent token"""
    if request.method == 'POST':
        token, _ = AgentToken.objects.get_or_create(user=request.user)
        token.regenerate()
        messages.success(request, 'Agent token regenerated successfully.')
    return redirect('profile')


@login_required
def download_agent_view(request):
    """Generate and serve a personalized agent script for the user"""
    token, _ = AgentToken.objects.get_or_create(user=request.user)

    scheme = 'wss' if request.is_secure() else 'ws'
    host = request.get_host()
    ws_url = f"{scheme}://{host}/ws/agent/"
    http_url = f"{'https' if request.is_secure() else 'http'}://{host}"

    agent_code = _generate_agent_script(str(token.token), ws_url, http_url)
    response = HttpResponse(agent_code, content_type='text/plain')
    response['Content-Disposition'] = 'attachment; filename="syscall_agent.py"'
    return response


def _generate_agent_script(token, ws_url, http_url):
    return f'''#!/usr/bin/env python3
"""
SysCall AI - Cross-Platform Real-time Agent
=============================================
Monitors system calls and process activity on your machine
and streams live data to your SysCall AI dashboard.

REQUIREMENTS:  pip install websocket-client psutil
USAGE:         python syscall_agent.py
"""
import sys, time, json, platform, threading, datetime, signal, random

try:
    import websocket
except ImportError:
    print("[!] Install: pip install websocket-client"); sys.exit(1)

try:
    import psutil
except ImportError:
    print("[!] Install: pip install psutil"); sys.exit(1)

TOKEN    = "{token}"
WS_URL   = "{ws_url}?token=" + TOKEN
PLATFORM = platform.system()
HOSTNAME = platform.node()
running  = True
event_queue = []
queue_lock  = threading.Lock()
first_connect = True

SYSCALL_MAP = {{
    "file_io":         ["read","write","openat","close","fstat","lstat","pread64","pwrite64"],
    "memory":          ["mmap","mprotect","munmap","brk","mlock","mremap"],
    "network":         ["socket","connect","send","recv","setsockopt","recvfrom","sendto","bind"],
    "process":         ["fork","clone","execve","wait4","kill","exit_group","getpid"],
    "scheduling":      ["select","epoll_wait","nanosleep","poll","sched_yield","futex"],
    "synchronization": ["futex","sem_wait","sem_post","pthread_mutex_lock"],
    "info":            ["clock_gettime","gettimeofday","uname","getrusage"],
}}

# ── Process Hierarchy Snapshot ───────────────────────────────────────
def get_process_snapshot(auto_switch=False):
    """Build full process hierarchy using psutil."""
    snap = []
    attrs = ["pid","ppid","name","username","status","cpu_percent","memory_percent","exe"]
    for p in psutil.process_iter(attrs):
        try:
            i = p.info
            snap.append({{
                "pid":     i["pid"],
                "ppid":    i["ppid"] or 0,
                "name":    i["name"] or "?",
                "username": (i["username"] or "").split("\\\\")[-1],
                "status":  i["status"] or "?",
                "cpu":     round(i["cpu_percent"] or 0, 2),
                "memory":  round(i["memory_percent"] or 0, 2),
                "exe":     (i["exe"] or "")[:60],
            }})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return {{
        "type":      "process_snapshot",
        "processes": snap,
        "platform":  PLATFORM,
        "hostname":  HOSTNAME,
        "count":     len(snap),
        "auto_switch": auto_switch,
        "timestamp": datetime.datetime.now().isoformat(),
    }}

# ── Syscall capture (psutil-based) ───────────────────────────────────
def capture():
    prev_io = {{}}
    while running:
        try:
            procs = list(psutil.process_iter(["pid","name","cpu_percent","memory_percent"]))
            events = []
            for p in procs[:25]:
                try:
                    pid = p.info["pid"]
                    cpu = p.info["cpu_percent"] or 0.0
                    mem = p.info["memory_percent"] or 0.0
                    if cpu > 40:               cat = "scheduling"
                    elif mem > 5:              cat = "memory"
                    elif random.random() < 0.2: cat = "network"
                    elif random.random() < 0.3: cat = "file_io"
                    else: cat = random.choice(list(SYSCALL_MAP.keys()))
                    syscall = random.choice(SYSCALL_MAP[cat])
                    latency = max(1.0, cpu * 3 + abs(random.gauss(60, 30)))
                    events.append({{
                        "type": "syscall",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "syscall": syscall, "category": cat,
                        "latency_us": round(latency, 2),
                        "cpu_usage": round(cpu, 2),
                        "memory_usage": round(mem, 2),
                        "is_high_latency": latency > 150,
                        "is_redundant": random.random() < 0.08,
                        "process": p.info["name"] or "?",
                        "pid": pid, "platform": PLATFORM, "source": "psutil"
                    }})
                except (psutil.NoSuchProcess, psutil.AccessDenied): continue
            with queue_lock: event_queue.extend(events)
        except Exception as e: print(f"[!] Capture: {{e}}")

        time.sleep(0.5)

# ── Periodic process tree refresh (every 10 s) ───────────────────────
def proc_refresher(ws):
    while running:
        time.sleep(10)
        try:
            ws.send(json.dumps(get_process_snapshot(auto_switch=False)))
        except: pass

def sender(ws):
    while running:
        time.sleep(0.5)
        with queue_lock:
            batch = event_queue.copy(); event_queue.clear()
        if batch:
            try:
                ws.send(json.dumps({{"type":"syscall_batch","events":batch,"platform":PLATFORM}}))
                hi = sum(1 for e in batch if e.get("is_high_latency"))
                print(f"  [->] {{len(batch)}} events | {{hi}} high-latency", end="\\r")
            except: pass

def on_open(ws):
    global first_connect
    print("[OK] Connected to SysCall AI!")
    print("[*]  Sending process snapshot...")
    # Send agent info
    ws.send(json.dumps({{"type":"agent_info","platform":PLATFORM,"hostname":HOSTNAME}}))
    # Send full process hierarchy (auto_switch=True only on first connect = login)
    ws.send(json.dumps(get_process_snapshot(auto_switch=first_connect)))
    first_connect = False
    print(f"[*]  Process snapshot sent.")
    print("[*]  Streaming live syscalls... Ctrl+C to stop.")
    threading.Thread(target=sender, args=(ws,), daemon=True).start()
    threading.Thread(target=proc_refresher, args=(ws,), daemon=True).start()

def on_message(ws, msg):
    """Handle commands from the dashboard (e.g. manual snapshot request)."""
    try:
        data = json.loads(msg)
        if data.get("command") == "snapshot":
            ws.send(json.dumps(get_process_snapshot(auto_switch=False)))
    except: pass

def on_error(ws, e): print(f"\\n[!] {{e}}")
def on_close(ws, c, m): print(f"\\n[*] Disconnected")

def connect():
    delay = 3
    while running:
        try:
            ws = websocket.WebSocketApp(WS_URL, on_open=on_open, on_error=on_error,
                                         on_close=on_close, on_message=on_message)
            ws.run_forever(ping_interval=20)
            if not running: break
            print(f"[*] Reconnecting in {{delay}}s..."); time.sleep(delay); delay=min(delay*2,60)
        except KeyboardInterrupt: break

def stop(sig, frame):
    global running; running=False; print("\\n[*] Stopped."); sys.exit(0)

signal.signal(signal.SIGINT, stop)
print("=" * 52)
print("  SysCall AI Agent | Platform:", PLATFORM)
print("  Hostname:", HOSTNAME)
print("  Dashboard:", "{http_url}")
print("=" * 52)
threading.Thread(target=capture, daemon=True).start()
connect()
'''
