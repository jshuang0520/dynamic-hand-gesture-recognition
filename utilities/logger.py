import datetime

def get_logger(name="RESNET_LSTM_PIPELINE"):
    def log(message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{level}] {timestamp} | {name} | {message}")
    return log