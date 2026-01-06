"""
XAI Error Handling and Logging Configuration
Provides robust error handling and detailed logging for debugging
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_xai_logging(log_file: str = "xai.log", log_level: int = logging.INFO):
    """
    Configure logging for XAI system
    
    Args:
        log_file: Path to log file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('XAI')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (simple logs)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only show warnings and errors in console
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)
    
    logger.info("="*60)
    logger.info(f"XAI Logging initialized - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60)
    
    return logger


class XAIErrorHandler:
    """Centralized error handling for XAI operations"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger('XAI')
        self.error_counts = {
            'c1': 0,
            'c2': 0,
            'c3': 0,
            'llm': 0,
            'cache': 0
        }
    
    def handle_explanation_error(self, component: str, error: Exception, 
                                 context: str = "") -> dict:
        """
        Handle XAI explanation errors gracefully
        
        Args:
            component: Component name (c1, c2, c3)
            error: Exception that occurred
            context: Additional context
        
        Returns:
            Empty explanation dict (graceful degradation)
        """
        self.error_counts[component] += 1
        
        self.logger.error(
            f"XAI {component.upper()} Error ({self.error_counts[component]} total): {str(error)}",
            exc_info=True
        )
        
        if context:
            self.logger.debug(f"Context: {context}")
        
        # Return empty explanation
        return {
            'error': str(error),
            'component': component,
            'fallback': True
        }
    
    def handle_cache_error(self, error: Exception, operation: str = "get"):
        """Handle cache errors"""
        self.error_counts['cache'] += 1
        self.logger.warning(
            f"Cache {operation} failed: {error}. Continuing without cache."
        )
    
    def handle_llm_error(self, error: Exception, fallback_used: bool = True):
        """Handle LLM translation errors"""
        self.error_counts['llm'] += 1
        
        if fallback_used:
            self.logger.info(
                f"LLM unavailable: {error}. Using template explanations."
            )
        else:
            self.logger.error(
                f"LLM Error: {error}",
                exc_info=True
            )
    
    def get_error_summary(self) -> dict:
        """Get summary of all errors"""
        total_errors = sum(self.error_counts.values())
        return {
            'total_errors': total_errors,
            'by_component': self.error_counts.copy(),
            'has_errors': total_errors > 0
        }
    
    def log_performance(self, component: str, duration: float, cached: bool = False):
        """Log performance metrics"""
        cache_str = " (cached)" if cached else ""
        self.logger.info(f"{component.upper()} explanation took {duration:.2f}s{cache_str}")


# Example usage:
"""
# In XAI_enhanced.py:
from xai_error_handling import setup_xai_logging, XAIErrorHandler

# During initialization:
xai_logger = setup_xai_logging(log_file="xai.log", log_level=logging.INFO)
error_handler = XAIErrorHandler(xai_logger)

# When explaining:
import time

try:
    start = time.time()
    explanation = state['xai_explainer'].explain_c1(features, log)
    duration = time.time() - start
    error_handler.log_performance('c1', duration)
except Exception as e:
    explanation = error_handler.handle_explanation_error('c1', e, context=f"Log: {log}")
"""
