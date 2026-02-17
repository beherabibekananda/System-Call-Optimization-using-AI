from django.urls import path
from . import api_views

urlpatterns = [
    path("status", api_views.get_status, name="api_status"),
    path("strace/parse", api_views.parse_strace_file, name="api_parse_file"),
    path("strace/parse-text", api_views.parse_strace_text, name="api_parse_text"),
    path("strace/run", api_views.api_strace_run, name="api_strace_run"),
    path("strace/tool-info", api_views.get_trace_tool_info, name="api_tool_info"),
    path("strace/samples", api_views.get_strace_samples, name="api_samples"),
    path("strace/load-sample", api_views.load_strace_sample, name="api_load_sample"),
    path("export", api_views.export_report, name="api_export"),
    # Parity with previous Flask API
    path("analyze", api_views.api_analyze, name="api_analyze"),
    path("predict", api_views.api_predict, name="api_predict"),
    path("benchmark", api_views.api_benchmark, name="api_benchmark"),
]
