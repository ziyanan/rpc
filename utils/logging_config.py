import logging


class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[92m',
        'INFO': '\033[94m',
        'WARNING': '\033[93m',
        'ERROR': '\033[91m',
        'CRITICAL': '\033[95m',
    }
    RESET = '\033[0m'
    
    def format(self, record):
        original_format = super().format(record)
        
        level_name = record.levelname
        if level_name in self.COLORS:
            colored_format = f"{self.COLORS[level_name]}{original_format}{self.RESET}"
            return colored_format
        
        return original_format


def setup_colored_logging():
    colored_formatter = ColoredFormatter('%(name)s - %(levelname)s - %(message)s')
    file_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler('logs/smart_city_simulation.log')
    file_handler.setFormatter(file_formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(colored_formatter)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
