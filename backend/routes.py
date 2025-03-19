from flask import Blueprint

api = Blueprint('api', __name__)

@api.route('/')
def hello():
    return "Welcome to flask api!"

# This are some sample routes
@api.route('/users', methods=['GET'])
def get_users():
    return {"message": "List of users"}

@api.route('/posts', methods=['GET'])
def get_posts():
    return {"message": "List of posts"}
