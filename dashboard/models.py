from django.db import models
from django.contrib.auth.models import User
import json


class StraceAnalysis(models.Model):
    """Stores each user's strace analysis results"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='analyses')
    title = models.CharField(max_length=200)
    source_type = models.CharField(max_length=50, choices=[
        ('file', 'Uploaded File'),
        ('text', 'Pasted Text'),
        ('sample', 'Sample File'),
        ('live', 'Live Trace'),
        ('synthetic', 'Synthetic Data'),
    ])
    filename = models.CharField(max_length=200, blank=True, default='')
    uploaded_file = models.FileField(upload_to='strace_uploads/', blank=True, null=True)

    # Results (stored as JSON)
    total_syscalls = models.IntegerField(default=0)
    analysis_json = models.TextField(default='{}')
    recommendations_json = models.TextField(default='[]')
    syscall_summary_json = models.TextField(default='{}')
    category_summary_json = models.TextField(default='{}')
    parse_stats_json = models.TextField(default='{}')

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Strace Analyses'

    def __str__(self):
        return f"{self.user.username} - {self.title} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"

    @property
    def analysis(self):
        return json.loads(self.analysis_json)

    @property
    def recommendations(self):
        return json.loads(self.recommendations_json)

    @property
    def syscall_summary(self):
        return json.loads(self.syscall_summary_json)

    @property
    def category_summary(self):
        return json.loads(self.category_summary_json)

    @property
    def parse_stats(self):
        return json.loads(self.parse_stats_json)


class ExportHistory(models.Model):
    """Stores export history per user"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='exports')
    analysis = models.ForeignKey(StraceAnalysis, on_delete=models.SET_NULL, null=True, blank=True)
    export_format = models.CharField(max_length=20, choices=[
        ('script', 'Shell Script'),
        ('json', 'JSON Report'),
        ('markdown', 'Markdown Report'),
    ])
    filename = models.CharField(max_length=200)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name_plural = 'Export Histories'

    def __str__(self):
        return f"{self.user.username} - {self.export_format} ({self.created_at.strftime('%Y-%m-%d %H:%M')})"
