from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import logging
from datetime import datetime
import os

try:
    from app.core.config import settings
    from app.core.database import engine, Base
    from app.api import routes_cml, routes_forecast, routes_report, routes_dashboard
except ImportError:
    # Fallback for initial setup
    settings = type('obj', (object,), {'ALLOWED_ORIGINS': 'http://localhost:3000,http://localhost:8000', 'DEBUG': True})()
    engine = None
    Base = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Wood AI CML Optimization",
    description="Machine Learning system for Condition Monitoring Location optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS.split(',') if hasattr(settings, 'ALLOWED_ORIGINS') else ['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Wood AI CML Optimization API...")
    if Base and engine:
        try:
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {e}")
    else:
        logger.warning("Database not configured - running in standalone mode")

# Include API routers (with error handling for initial setup)
try:
    app.include_router(routes_cml.router, prefix="/api/v1/cml", tags=["CML Operations"])
    app.include_router(routes_forecast.router, prefix="/api/v1/forecast", tags=["Forecasting"])
    app.include_router(routes_report.router, prefix="/api/v1/report", tags=["Reports"])
    app.include_router(routes_dashboard.router, prefix="/api/v1/dashboard", tags=["Dashboard"])
    logger.info("API routes loaded successfully")
except Exception as e:
    logger.warning(f"Some routes not available yet: {e}")

# Root endpoint with beautiful dashboard
@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Wood AI CML Optimization</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header p {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        .status-bar {{
            background: #d4edda;
            border: 2px solid #c3e6cb;
            color: #155724;
            padding: 15px 20px;
            margin: 20px 40px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
        }}
        .status-item {{
            margin: 5px 10px;
        }}
        .status-item strong {{
            margin-right: 5px;
        }}
        .content {{
            padding: 40px;
        }}
        .features {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .feature-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 15px;
            transition: transform 0.3s, box-shadow 0.3s;
            cursor: pointer;
        }}
        .feature-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.4);
        }}
        .feature-card h3 {{
            font-size: 1.4em;
            margin-bottom: 10px;
        }}
        .feature-card p {{
            opacity: 0.95;
            line-height: 1.5;
        }}
        .links {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        .links h3 {{
            color: #667eea;
            margin-bottom: 20px;
        }}
        .link-buttons {{
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .link-buttons a {{
            display: inline-block;
            padding: 12px 24px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            transition: background 0.3s, transform 0.2s;
            font-weight: 500;
        }}
        .link-buttons a:hover {{
            background: #764ba2;
            transform: translateY(-2px);
        }}
        .getting-started {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            color: #856404;
            padding: 25px;
            border-radius: 15px;
            margin: 30px 0;
        }}
        .getting-started h3 {{
            margin-bottom: 15px;
            color: #856404;
        }}
        .getting-started ol {{
            margin-left: 20px;
        }}
        .getting-started li {{
            margin: 10px 0;
            line-height: 1.6;
        }}
        .getting-started code {{
            background: #fff;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 Wood AI CML Optimization</h1>
            <p>Machine Learning for Condition Monitoring Location Analysis</p>
        </div>
        
        <div class="status-bar">
            <div class="status-item"><strong>✅ Status:</strong> Running</div>
            <div class="status-item"><strong>📚 Version:</strong> 1.0.0</div>
            <div class="status-item"><strong>⏰ Time:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
        </div>
        
        <div class="content">
            <div class="features">
                <div class="feature-card">
                    <h3>🤖 ML Predictions</h3>
                    <p>XGBoost-powered CML elimination recommendations with SHAP explanations for transparency</p>
                </div>
                <div class="feature-card">
                    <h3>📈 Forecasting</h3>
                    <p>Time-series prediction of remaining asset life using Prophet and statistical models</p>
                </div>
                <div class="feature-card">
                    <h3>📊 Dashboard</h3>
                    <p>Interactive visualization of risk levels, corrosion trends, and real-time analytics</p>
                </div>
                <div class="feature-card">
                    <h3>📄 PDF Reports</h3>
                    <p>Automated client-ready inspection reports with actionable recommendations</p>
                </div>
            </div>
            
            <div class="links">
                <h3>📚 Quick Links</h3>
                <div class="link-buttons">
                    <a href="/docs" target="_blank">API Documentation</a>
                    <a href="/redoc" target="_blank">ReDoc</a>
                    <a href="/health" target="_blank">Health Check</a>
                    <a href="https://github.com/aaron-seq/Wood-AI-CML-ALO-Machine-Learning-Model" target="_blank">GitHub</a>
                </div>
            </div>
            
            <div class="getting-started">
                <h3>🚀 Getting Started</h3>
                <ol>
                    <li>Upload CML data via <code>POST /api/v1/cml/upload</code></li>
                    <li>Run ML analysis using <code>POST /api/v1/cml/analyze</code></li>
                    <li>View forecasts at <code>GET /api/v1/forecast/{{cml_id}}</code></li>
                    <li>Generate reports via <code>GET /api/v1/report/generate</code></li>
                    <li>Check dashboard metrics at <code>GET /api/v1/dashboard/summary</code></li>
                </ol>
            </div>
        </div>
        
        <div class="footer">
            <p>© 2025 Wood Engineering | Powered by FastAPI & ML | Developed by Aaron Sequeira</p>
        </div>
    </div>
</body>
</html>
"""
    return html_content

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "Wood AI CML Optimization API",
        "database": "connected" if engine else "not configured"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
