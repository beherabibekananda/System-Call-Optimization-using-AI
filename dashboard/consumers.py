"""
WebSocket Consumers for SysCall AI Real-time Monitoring
"""

import json
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.utils import timezone


class SyscallMonitorConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for the browser dashboard.
    Joins a group for the logged-in user and displays incoming syscall events.
    """

    async def connect(self):
        self.user = self.scope["user"]
        if not self.user.is_authenticated:
            await self.close()
            return

        self.room_group_name = f"syscall_user_{self.user.id}"

        # Join user-specific group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

        # Send welcome message
        await self.send(text_data=json.dumps({
            "type": "connection_established",
            "message": f"Connected! Waiting for agent data from your machine...",
            "user": self.user.username,
            "group": self.room_group_name
        }))

    async def disconnect(self, close_code):
        if hasattr(self, 'room_group_name'):
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )

    async def receive(self, text_data):
        """Handle messages from browser — forward snapshot requests to agent"""
        try:
            data = json.loads(text_data)
            cmd = data.get("command")
            if cmd == "ping":
                await self.send(text_data=json.dumps({"type": "pong"}))
            elif cmd == "snapshot":
                # Broadcast snapshot request to the agent group
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {"type": "request_snapshot", "data": {"command": "snapshot"}}
                )
        except Exception:
            pass

    async def syscall_event(self, event):
        """Send syscall event data to the browser"""
        await self.send(text_data=json.dumps(event["data"]))

    async def request_snapshot(self, event):
        """Ignored on browser consumer — agent consumer handles this"""
        pass



class AgentConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for the local agent (runs on user's machine).
    Receives live syscall data from the agent and broadcasts to the user's browser session.
    """

    async def connect(self):
        # The agent sends a token in the URL query param: /ws/agent/?token=<user_token>
        query_string = self.scope.get("query_string", b"").decode()
        self.user_token = None
        self.user_id = None

        for param in query_string.split("&"):
            if param.startswith("token="):
                self.user_token = param.split("=", 1)[1]
                break

        # Validate token
        user = await self.get_user_from_token(self.user_token)
        if not user:
            await self.close(code=4001)
            return

        self.agent_user = user
        self.room_group_name = f"syscall_user_{user.id}"

        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

        await self.send(text_data=json.dumps({
            "type": "agent_connected",
            "message": f"Agent authenticated for user: {user.username}"
        }))

        # Broadcast to dashboard that agent is online
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                "type": "syscall_event",
                "data": {
                    "type": "agent_online",
                    "user": user.username,
                    "platform": "unknown",
                    "timestamp": timezone.now().isoformat()
                }
            }
        )

    async def disconnect(self, close_code):
        if hasattr(self, 'room_group_name'):
            # Notify dashboard the agent went offline
            try:
                await self.channel_layer.group_send(
                    self.room_group_name,
                    {
                        "type": "syscall_event",
                        "data": {
                            "type": "agent_offline",
                            "timestamp": timezone.now().isoformat()
                        }
                    }
                )
            except Exception:
                pass
            await self.channel_layer.group_discard(
                self.room_group_name,
                self.channel_name
            )

    async def receive(self, text_data):
        """Receive syscall data from the agent and relay to browser"""
        try:
            data = json.loads(text_data)
            event_type = data.get("type", "syscall")

            if event_type == "heartbeat":
                await self.send(text_data=json.dumps({"type": "heartbeat_ack"}))
                return

            # Broadcast to the user's browser group
            await self.channel_layer.group_send(
                self.room_group_name,
                {
                    "type": "syscall_event",
                    "data": data
                }
            )
        except json.JSONDecodeError:
            pass
        except Exception as e:
            await self.send(text_data=json.dumps({"type": "error", "message": str(e)}))

    async def syscall_event(self, event):
        """Forward group messages back to agent (if needed)"""
        pass

    async def request_snapshot(self, event):
        """Send snapshot command to the agent"""
        try:
            await self.send(text_data=json.dumps(event["data"]))
        except Exception:
            pass

    @database_sync_to_async
    def get_user_from_token(self, token):
        """Validate agent token and return user"""
        if not token:
            return None
        try:
            from accounts.models import AgentToken
            agent_token = AgentToken.objects.select_related("user").get(token=token, is_active=True)
            return agent_token.user
        except Exception:
            return None
