import urllib.request
try:
    urllib.request.urlopen("https://syscall-ai-platform.vercel.app/")
except Exception as e:
    print(e)
