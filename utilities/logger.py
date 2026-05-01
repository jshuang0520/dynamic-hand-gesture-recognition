import datetime

def get_logger(name="SHAN_PIPELINE"):
    """Returns a custom logger function to prevent shadowing the math.log module."""
    def custom_logger(message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{level}] {timestamp} | {name} | {message}")
    return custom_logger