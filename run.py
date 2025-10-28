from src import create_app
from config import Config

app = create_app(Config)

if __name__ == "__main__":
    app.run(debug=Config.DEBUG, host="0.0.0.0", port=Config.PORT)
