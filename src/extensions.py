from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_swagger_ui import get_swaggerui_blueprint

cors = CORS()
jwt = JWTManager()

swaggerui_bp = get_swaggerui_blueprint(
    "/swagger",
    "/static/swagger.json",
    config={"app_name": "Sardine's API"},
)
