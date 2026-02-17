from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from .models import StraceAnalysis, ExportHistory


@login_required
def dashboard_view(request):
    """Main dashboard - the SysCall AI platform"""
    user_analyses = StraceAnalysis.objects.filter(user=request.user)[:10]
    user_exports = ExportHistory.objects.filter(user=request.user)[:5]

    context = {
        'analyses': user_analyses,
        'exports': user_exports,
        'total_analyses': user_analyses.count(),
    }
    return render(request, 'dashboard/index.html', context)


@login_required
def history_view(request):
    """View analysis history"""
    analyses = StraceAnalysis.objects.filter(user=request.user)
    return render(request, 'dashboard/history.html', {'analyses': analyses})
