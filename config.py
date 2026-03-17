

import os

# 資料庫連接設定
MYSQL_CONFIG = {
    'host': '192.168.150.136',
    'port': 3306,
    'user': 'root',
    'password': '12345678',
    'database': 'early_intervention_db'
}

NEO4J_CONFIG = {
    'uri': 'bolt://192.168.150.136:7687',
    'user': 'neo4j',
    'password': 'password'
}

LLM_CONFIG = {
    'base_url': 'http://192.168.150.136:8000/v1',
    'api_key': 'vllm-key',
    'model': 'google/gemma-3-4b-it'
}

EMBED_CONFIG = {
    'url': 'http://192.168.150.136:8080/embed'
}

# 應用程式設定
SECRET_KEY = 'your-secret-key-here'
UPLOAD_FOLDER = 'uploads'

# 從環境變數覆蓋設定（如果存在）
MYSQL_CONFIG['host'] = os.environ.get('MYSQL_HOST', MYSQL_CONFIG['host'])
MYSQL_CONFIG['port'] = int(os.environ.get('MYSQL_PORT', MYSQL_CONFIG['port']))
MYSQL_CONFIG['user'] = os.environ.get('MYSQL_USER', MYSQL_CONFIG['user'])
MYSQL_CONFIG['password'] = os.environ.get('MYSQL_PASSWORD', MYSQL_CONFIG['password'])
MYSQL_CONFIG['database'] = os.environ.get('MYSQL_DATABASE', MYSQL_CONFIG['database'])

NEO4J_CONFIG['uri'] = os.environ.get('NEO4J_URI', NEO4J_CONFIG['uri'])
NEO4J_CONFIG['user'] = os.environ.get('NEO4J_USER', NEO4J_CONFIG['user'])
NEO4J_CONFIG['password'] = os.environ.get('NEO4J_PASSWORD', NEO4J_CONFIG['password'])

# 生成資料庫 URI
def get_mysql_uri():
    """生成 MySQL 連接 URI"""
    return f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}"

def get_neo4j_uri():
    """獲取 Neo4j 連接 URI"""
    return NEO4J_CONFIG['uri']

def get_neo4j_auth():
    """獲取 Neo4j 認證資訊"""
    return (NEO4J_CONFIG['user'], NEO4J_CONFIG['password'])

