import json
import os
import pandas as pd
import numpy as np
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.conf import settings
from .models import StraceAnalysis, ExportHistory

# Import existing ML and Strace processing logic
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'backend'))

from ml_models.syscall_analyzer import SyscallAnalyzer
from strace.strace_parser import StraceParser
from strace.strace_runner import StraceRunner
from strace.script_exporter import ScriptExporter

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return obj.isoformat()
        if isinstance(obj, (pd.Series, pd.Index, np.ndarray)):
            return obj.tolist()
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        if isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        if isinstance(obj, (np.void)): 
            return None
        return json.JSONEncoder.default(self, obj)

def json_res(data, status=200):
    """Helper to return JSON with numpy support"""
    return HttpResponse(
        json.dumps(data, cls=NumpyEncoder),
        content_type='application/json',
        status=status
    )

# ... (rest of the code)
# (In a production Django app, these might be handled by signals or AppConfig)
analyzer = SyscallAnalyzer()
strace_parser = StraceParser()
strace_runner = StraceRunner()
script_exporter = ScriptExporter()

def ensure_models_trained():
    # Similar logic to Flask's ensure_models_trained if necessary
    # For now, we assume the .pkl files exist from previous sessions
    pass

@csrf_exempt
@login_required
def parse_strace_file(request):
    """Parse an uploaded strace output file and save it to the user's account"""
    if request.method != 'POST':
        return json_res({'success': False, 'error': 'Only POST allowed'}, status=405)
    
    try:
        if 'file' not in request.FILES:
            return json_res({'success': False, 'error': 'No file uploaded'}, status=400)
        
        file = request.FILES['file']
        
        # Save analysis record
        analysis_record = StraceAnalysis.objects.create(
            user=request.user,
            title=f"FileUpload_{file.name}",
            source_type='file',
            filename=file.name,
            uploaded_file=file
        )
        
        # Parse the file
        filepath = analysis_record.uploaded_file.path
        df = strace_parser.parse_file(filepath)
        parse_stats = strace_parser.get_parse_stats()
        
        if df.empty:
            return json_res({
                'success': False,
                'error': 'No syscalls could be parsed from the file',
                'parse_stats': parse_stats
            }, status=400)
        
        # Run ML analysis
        recommendations, analysis = analyzer.get_optimization_recommendations(df)
        
        # Save results to the record
        analysis_record.total_syscalls = len(df)
        analysis_record.analysis_json = json.dumps(analysis)
        analysis_record.recommendations_json = json.dumps(recommendations)
        analysis_record.syscall_summary_json = json.dumps(df['syscall_name'].value_counts().head(15).to_dict())
        analysis_record.category_summary_json = json.dumps(df['category'].value_counts().to_dict())
        analysis_record.parse_stats_json = json.dumps(parse_stats)
        analysis_record.save()
        
        return json_res({
            'success': True,
            'id': analysis_record.id,
            'parse_stats': parse_stats,
            'total_syscalls': len(df),
            'analysis': analysis,
            'recommendations': recommendations,
            'syscall_summary': json.loads(analysis_record.syscall_summary_json),
            'category_summary': json.loads(analysis_record.category_summary_json),
            'source': 'strace_file',
            'filename': file.name
        })
    except Exception as e:
        return json_res({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
def parse_strace_text(request):
    """Parse pasted strace text output"""
    if request.method != 'POST':
        return json_res({'success': False, 'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        strace_text = data.get('text', '')
        
        if not strace_text.strip():
            return json_res({'success': False, 'error': 'No strace text provided'}, status=400)
        
        # Parse the text
        df = strace_parser.parse_text(strace_text)
        parse_stats = strace_parser.get_parse_stats()
        
        if df.empty:
            return json_res({
                'success': False,
                'error': 'No syscalls could be parsed from the text',
                'parse_stats': parse_stats
            }, status=400)
        
        # Run ML analysis
        recommendations, analysis = analyzer.get_optimization_recommendations(df)
        
        # Save analysis record
        analysis_record = StraceAnalysis.objects.create(
            user=request.user,
            title="PastedTextAnalysis",
            source_type='text',
            total_syscalls=len(df),
            analysis_json=json.dumps(analysis),
            recommendations_json=json.dumps(recommendations),
            syscall_summary_json=json.dumps(df['syscall_name'].value_counts().head(15).to_dict()),
            category_summary_json=json.dumps(df['category'].value_counts().to_dict()),
            parse_stats_json=json.dumps(parse_stats)
        )
        
        return json_res({
            'success': True,
            'id': analysis_record.id,
            'parse_stats': parse_stats,
            'total_syscalls': len(df),
            'analysis': analysis,
            'recommendations': recommendations,
            'syscall_summary': json.loads(analysis_record.syscall_summary_json),
            'category_summary': json.loads(analysis_record.category_summary_json),
            'source': 'strace_text'
        })
    except Exception as e:
        return json_res({'success': False, 'error': str(e)}, status=500)

@login_required
def get_strace_samples(request):
    """List available sample strace files"""
    SAMPLES_PATH = os.path.join(BASE_DIR, 'backend', 'strace', 'samples')
    samples = []
    if os.path.exists(SAMPLES_PATH):
        for f in os.listdir(SAMPLES_PATH):
            if f.endswith('.strace'):
                samples.append({
                    'filename': f,
                    'size': os.path.getsize(os.path.join(SAMPLES_PATH, f)),
                    'description': f.replace('.strace', '').replace('_', ' ').title()
                })
    return json_res({'success': True, 'samples': samples})

@csrf_exempt
@login_required
def load_strace_sample(request):
    """Load and parse a sample strace file"""
    if request.method != 'POST':
        return json_res({'success': False, 'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        filename = data.get('filename', '')
        
        SAMPLES_PATH = os.path.join(BASE_DIR, 'backend', 'strace', 'samples')
        filepath = os.path.join(SAMPLES_PATH, filename)
        if not os.path.exists(filepath):
            return json_res({'success': False, 'error': f'Sample not found: {filename}'}, status=404)
        
        # Parse the sample
        df = strace_parser.parse_file(filepath)
        parse_stats = strace_parser.get_parse_stats()
        
        recommendations, analysis = analyzer.get_optimization_recommendations(df)
        
        return json_res({
            'success': True,
            'parse_stats': parse_stats,
            'total_syscalls': len(df),
            'analysis': analysis,
            'recommendations': recommendations,
            'syscall_summary': df['syscall_name'].value_counts().head(15).to_dict(),
            'category_summary': df['category'].value_counts().to_dict(),
            'source': 'strace_sample',
            'filename': filename
        })
    except Exception as e:
        return json_res({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
def export_report(request):
    """Export recommendations as shell script, JSON, or Markdown and save to user history"""
    if request.method != 'POST':
        return json_res({'success': False, 'error': 'Only POST allowed'}, status=405)
        
    try:
        data = json.loads(request.body)
        export_format = data.get('format', 'script')
        recommendations = data.get('recommendations', [])
        analysis = data.get('analysis', {})
        strace_stats = data.get('strace_stats', {})
        
        if export_format == 'script':
            result = script_exporter.export_optimization_script(recommendations, analysis)
        elif export_format == 'json':
            result = script_exporter.export_json_report(recommendations, analysis, strace_stats)
        elif export_format == 'markdown':
            result = script_exporter.export_markdown_report(recommendations, analysis, strace_stats)
        else:
            return json_res({'success': False, 'error': f'Unknown format: {export_format}'}, status=400)
        
        # Save to ExportHistory
        ExportHistory.objects.create(
            user=request.user,
            export_format=export_format,
            filename=result['filename'],
            content=result.get('content', '')
        )
        
        return json_res(result)
    except Exception as e:
        return json_res({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
def get_status(request):
    """Get overall platform status for the current user"""
    analyses_count = StraceAnalysis.objects.filter(user=request.user).count()
    return json_res({
        'status': 'active',
        'user': request.user.username,
        'analyses_count': analyses_count
    })

@csrf_exempt
@login_required
def api_analyze(request):
    """Run a quick syscall analysis (synthetic or live)"""
    if request.method != 'POST':
        return json_res({'success': False, 'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        num_calls = data.get('num_calls', 1000)
        process_type = data.get('process_type', 'mixed')
        
        # In a real app, this would use live tracing or historical data
        # For the quick dashboard view, we generate some synthetic data
        from data.syscall_data_generator import SyscallDataGenerator
        generator = SyscallDataGenerator()
        df = generator.generate_syscall_sequence(num_calls, process_type)
        
        recommendations, analysis = analyzer.get_optimization_recommendations(df)
        
        return json_res({
            'success': True,
            'analysis': analysis,
            'recommendations': recommendations
        })
    except Exception as e:
        return json_res({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
def api_predict(request):
    """Run performance prediction and resource optimization"""
    if request.method != 'POST':
        return json_res({'success': False, 'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        from ml_models.performance_predictor import PerformancePredictor
        predictor = PerformancePredictor()
        
        results = predictor.generate_optimization_plan(
            num_processes=data.get('num_processes', 5),
            calls_per_process=data.get('calls_per_process', 500)
        )
        
        return json_res({
            'success': True,
            'recommendations': results
        })
    except Exception as e:
        return json_res({'success': False, 'error': str(e)}, status=500)

@csrf_exempt
@login_required
def api_benchmark(request):
    """Run performance benchmarking"""
    if request.method != 'POST':
        return json_res({'success': False, 'error': 'Only POST allowed'}, status=405)
    
    try:
        data = json.loads(request.body)
        iterations = data.get('iterations', 5)
        
        from benchmarks.benchmark_framework import SyscallBenchmark
        bench = SyscallBenchmark()
        results = bench.run_simulated_benchmark(iterations=iterations)
        
        return json_res({
            'success': True,
            'iterations': results['iterations'],
            'average_improvement': results['average_improvement']
        })
    except Exception as e:
        return json_res({'success': False, 'error': str(e)}, status=500)

