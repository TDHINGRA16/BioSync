"""
Alice Search Engine - Log Spam Suppressor
Kills all the noisy third-party error spam
"""

import logging
import warnings
import asyncio
import sys
from datetime import datetime

class CleanFormatter(logging.Formatter):
    """Clean, readable log formatter"""
    
    def format(self, record):
        timestamp = datetime.fromtimestamp(record.created).strftime('%H:%M:%S')
        
        # Clean up module names
        if record.name.startswith('utils.search.'):
            module = record.name.split('.')[-1]
            prefix = f"[{timestamp}] {module.upper()}:"
        elif record.name in ['api.search', '__main__', 'main']:
            prefix = f"[{timestamp}] API:"
        else:
            prefix = f"[{timestamp}]"
        
        # Format by level
        if record.levelname == 'INFO':
            return f"{prefix} {record.getMessage()}"
        elif record.levelname == 'WARNING':
            return f"{prefix} ⚠️  {record.getMessage()}"
        elif record.levelname == 'ERROR':
            return f"{prefix} ❌ {record.getMessage()}"
        else:
            return f"{prefix} {record.getMessage()}"

def suppress_all_spam():
    """Nuclear option: Suppress ALL third-party error spam"""
    
    # 1. Suppress asyncio errors (Playwright subprocess spam)
    def silent_exception_handler(loop, context): 
        """Silent handler for asyncio errors"""
        exception = context.get('exception')
        if exception:
            # Suppress these specific errors completely
            if isinstance(exception, (NotImplementedError, ConnectionError, OSError)):
                return
            if 'playwright' in str(exception).lower():
                return
            if 'subprocess' in str(exception).lower():
                return
        # Only log really important errors
        if context.get('message') and 'critical' in context.get('message', '').lower():
            logging.getLogger('asyncio').error(f"Critical async error: {context.get('message', 'Unknown')}")
    
    # Set the silent handler
    try:
        loop = asyncio.get_event_loop()
        loop.set_exception_handler(silent_exception_handler)
    except RuntimeError:
        pass
    
    # 2. Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # 3. Nuclear suppression of specific loggers
    spam_loggers = [
        'asyncio',
        'playwright', 
        'playwright._impl',
        'playwright._impl.connection',
        'playwright._impl.transport',
        'crawl4ai',
        'crawl4ai.async_crawler_strategy',
        'ddgs',
        'ddgs.ddgs',
        'primp',
        'httpx',
        'uvicorn',
        'uvicorn.access',
        'uvicorn.error',
        'multiprocessing'
    ]
    
    for logger_name in spam_loggers:
        spam_logger = logging.getLogger(logger_name)
        spam_logger.setLevel(logging.CRITICAL)  # Only show CRITICAL errors
        spam_logger.propagate = False  # Don't propagate to parent loggers
    
    # 4. Override root logger to catch any remaining spam
    class SpamFilter(logging.Filter):
        def filter(self, record):
            # Block specific error patterns
            message = record.getMessage().lower()
            spam_patterns = [
                'notimplementederror',
                'subprocess_exec', 
                'playwright_impl',
                'crawl4ai_scraper',
                'task exception was never retrieved',
                'http request failed',
                'connection.py',
                'transport.py',
                'async_crawler_strategy.py'
            ]
            
            for pattern in spam_patterns:
                if pattern in message:
                    return False
            
            return True
    
    # Add spam filter to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(SpamFilter())

def setup_clean_logging():
    """Setup clean logging with spam suppression"""
    
    # Remove all existing handlers
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Setup clean console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CleanFormatter())
    
    # Configure root logger
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(console_handler)
    
    # Allow only our loggers to show INFO
    our_loggers = [
        'api.search',
        'utils.search.scraper',
        'utils.search.search_engine', 
        'utils.search.llm_ranker',
        'utils.search.llm_organiser',
        'main',
        '__main__'
    ]
    
    for logger_name in our_loggers:
        our_logger = logging.getLogger(logger_name)
        our_logger.setLevel(logging.INFO)
    
    # Nuclear spam suppression
    suppress_all_spam()
    
    print("🧹 Log spam suppression activated - clean logs only!")

# Auto-setup when imported
setup_clean_logging()
