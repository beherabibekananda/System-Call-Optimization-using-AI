from django.contrib import admin
from .models import StraceAnalysis, ExportHistory


@admin.register(StraceAnalysis)
class StraceAnalysisAdmin(admin.ModelAdmin):
    list_display = ['user', 'title', 'source_type', 'total_syscalls', 'created_at']
    list_filter = ['source_type', 'created_at', 'user']
    search_fields = ['title', 'user__username']
    readonly_fields = ['created_at', 'updated_at']


@admin.register(ExportHistory)
class ExportHistoryAdmin(admin.ModelAdmin):
    list_display = ['user', 'export_format', 'filename', 'created_at']
    list_filter = ['export_format', 'created_at', 'user']
