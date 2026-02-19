import uuid
from django.db import models
from django.contrib.auth.models import User


class AgentToken(models.Model):
    """Token used by the local agent to authenticate with the WebSocket server"""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='agent_token')
    token = models.UUIDField(default=uuid.uuid4, unique=True, db_index=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_seen = models.DateTimeField(null=True, blank=True)
    platform = models.CharField(max_length=50, blank=True)  # "mac", "windows", "linux"

    def __str__(self):
        return f"Token for {self.user.username}"

    def regenerate(self):
        self.token = uuid.uuid4()
        self.save()
        return self.token
