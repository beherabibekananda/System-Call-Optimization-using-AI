from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    # Browser dashboard connects here
    re_path(r"ws/monitor/$", consumers.SyscallMonitorConsumer.as_asgi()),
    # Agent connects here with ?token=<token>
    re_path(r"ws/agent/$", consumers.AgentConsumer.as_asgi()),
]
