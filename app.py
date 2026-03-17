from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, abort
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from sqlalchemy import text
import os, json, traceback
from datetime import datetime

# 導入 pdf_parser 的處理流程
from pdf_parser.pdf_processor_main import IEPPipeline

# 導入對話狀態模組
from dialogue_state_module.embedding import TextEncoder, EncoderConfig, encode_anchors, encode_overview_anchors
from dialogue_state_module.domain_router import DomainRouter, DomainRouterConfig
from dialogue_state_module.semantic_flow_module_v2 import SemanticFlowClassifier
from dialogue_state_module.domain_anchors import load_domain_anchors
from dialogue_state_module.task_scope_classifier import TaskScopeClassifier


# 導入檢索組件
from retrieval_module_v2.graph_client import GraphClient

# 導入新的檢索模組 V2
from retrieval_module_v2 import RetrievalModuleV2

# 導入 LLM 生成模組
from llm_generate_module import LLMGenerator, LLMConfig, LLMPromptManager

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
from config import get_mysql_uri
app.config['SQLALCHEMY_DATABASE_URI'] = get_mysql_uri()
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'

from config import get_neo4j_uri, get_neo4j_auth
app.config['NEO4J_URI'] = get_neo4j_uri()
app.config['NEO4J_USER'] = get_neo4j_auth()[0]
app.config['NEO4J_PASSWORD'] = get_neo4j_auth()[1]

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 對話緩存目錄
DIALOGUE_CACHE_DIR = 'dialogue_cache'
os.makedirs(DIALOGUE_CACHE_DIR, exist_ok=True)

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# 資料庫模型
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Child(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    birth_date = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    caregiver_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    therapist_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    reports = db.relationship('Report', backref='child', lazy=True)
    caregiver = db.relationship('User', foreign_keys=[caregiver_id], backref='children')
    therapist = db.relationship('User', foreign_keys=[therapist_id], backref='created_children')

class Report(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)
    child_id = db.Column(db.Integer, db.ForeignKey('child.id'), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)

class KnowledgeGraphProcessing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    report_id = db.Column(db.Integer, db.ForeignKey('report.id'), nullable=False)
    doc_id = db.Column(db.String(100), nullable=False)
    total_chunks = db.Column(db.Integer, nullable=False)
    total_facets = db.Column(db.Integer, nullable=False)
    total_relationships = db.Column(db.Integer, nullable=False)
    neo4j_uri = db.Column(db.String(200), nullable=False)
    processing_time = db.Column(db.Float, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    error_message = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    report = db.relationship('Report', backref='kg_processing')

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    child_id = db.Column(db.Integer, db.ForeignKey('child.id'), nullable=True)
    message = db.Column(db.Text, nullable=False)
    is_user_message = db.Column(db.Boolean, nullable=False, default=True)
    sent_at = db.Column(db.DateTime, default=datetime.utcnow)
    # 對話狀態資訊
    flow_state = db.Column(db.Text, nullable=True)  # JSON 格式的對話狀態
    retrieval_info = db.Column(db.Text, nullable=True)  # JSON 格式的檢索資訊
    user = db.relationship('User', backref='chat_messages')
    child = db.relationship('Child', backref='chat_messages')

# 全局對話狀態追蹤模組變數
_shared_encoder = None  # TextEncoder - shared across all users (expensive to load)
_shared_domain_router = None  # DomainRouter - shared and stateless
_shared_anchor_vecs = None  # Cached anchor vectors
_shared_overview_anchor_vecs = None  # 整體意圖錨點向量（用於向量比、不用關鍵字）
_shared_task_scope_clf = None  # TaskScopeClassifier - shared across all users
_user_classifiers = {}  # key: (user_id, child_id) -> SemanticFlowClassifier instance

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

def init_dst_components():
    """
    Initialize shared DST components once at app startup.
    TextEncoder and DomainRouter are expensive to load, so we load them once
    and reuse across all users.
    """
    global _shared_encoder, _shared_domain_router, _shared_anchor_vecs, _shared_overview_anchor_vecs, _shared_task_scope_clf
    
    if _shared_encoder is not None:
        return  # Already initialized
    
    try:
        print("[DST] 初始化對話狀態追蹤模組...")
        
        # Initialize TextEncoder (uses external API server)
        encoder_cfg = EncoderConfig()
        _shared_encoder = TextEncoder(encoder_cfg)
        
        print("[DST] TextEncoder 已初始化（對接遠端 Embedding Server）")
        
        # Compute anchor vectors for all domains（從設定檔載入，失敗則用程式內建）
        _domains, _overview_anchors, _domain_anchors = load_domain_anchors()
        domain_cfg = DomainRouterConfig()
        _shared_anchor_vecs = encode_anchors(_shared_encoder, _domain_anchors, _domains)
        print("[DST] 域錨向量已計算")
        
        # 整體意圖錨點向量（用於向量比對判定整體，取代關鍵字）
        _shared_overview_anchor_vecs = encode_overview_anchors(_shared_encoder, _overview_anchors)
        print("[DST] 整體錨點向量已計算")
        
        # Initialize DomainRouter (stateless, references shared encoder)
        _shared_domain_router = DomainRouter(_shared_encoder, _domains, _shared_anchor_vecs, domain_cfg)
        print("[DST] DomainRouter 已初始化")
        
        # Initialize TaskScopeClassifier (shared, uses same encoder)
        _shared_task_scope_clf = TaskScopeClassifier(embedder=_shared_encoder)
        print("[DST] TaskScopeClassifier 已初始化（已啟用任務/範圍分類）")
        
        print("[DST] 對話狀態追蹤模組初始化完成！")
    except Exception as e:
        print(f"[DST] 初始化失敗: {e}")
        raise

def get_dialogue_classifier(user_id: int, child_id: int) -> SemanticFlowClassifier:
    """
    Get or create a SemanticFlowClassifier for a specific user/child pair.
    Each user/child conversation has its own classifier instance with separate
    memory (ContextSimilarity, MultiTopicTracker), but shares the expensive
    global TextEncoder and DomainRouter.
    
    Args:
        user_id: User ID
        child_id: Child ID
    
    Returns:
        SemanticFlowClassifier instance
    """
    global _user_classifiers, _shared_encoder, _shared_domain_router, _shared_task_scope_clf, _shared_overview_anchor_vecs
    
    if _shared_encoder is None:
        raise RuntimeError("DST components not initialized. Call init_dst_components() first.")
    
    key = (user_id, child_id)
    
    if key not in _user_classifiers:
        # Create new classifier instance for this user/child pair
        classifier = SemanticFlowClassifier(
            text_encoder=_shared_encoder,
            domain_router=_shared_domain_router,
            context_similarity=None,  # Will be created internally
            topic_tracker=None,  # Will be created internally
            policy_cfg=None,  # Will use default config
            enable_task_scope=True,  # 啟用任務/範圍分類
            task_scope_clf=_shared_task_scope_clf,  # 使用共用的 TaskScopeClassifier
            overview_anchor_vecs=_shared_overview_anchor_vecs,
            overview_sim_threshold=0.63,  # 整體意圖向量相似度門檻（僅在模糊時使用）
        )
        _user_classifiers[key] = classifier
        
        # 嘗試從文件加載之前的狀態（如果存在）
        if classifier.load_state(user_id, child_id):
            print(f"[DST] 為用戶 {user_id} 的兒童 {child_id} 恢復對話狀態（turn={classifier.turn_index}）")
        else:
            print(f"[DST] 為用戶 {user_id} 的兒童 {child_id} 創建新分類器（無歷史狀態）")
    
    return _user_classifiers[key]

def cleanup_inactive_classifiers(max_inactive_hours: int = 24):
    """
    Optional: Clean up classifiers that haven't been used recently.
    This prevents memory bloat for long-running servers with many users.
    """
    # Implementation: track last_used timestamp, remove if > max_inactive_hours
    # For now, this is a placeholder for future optimization
    pass

def process_report_after_upload(report_id: int, file_path: str, child_name: str):
    """
    上傳 PDF 後的自動處理流程
    使用 pdf_parser 模組處理 PDF 並插入 Neo4j
    
    Args:
        report_id: 報告 ID
        file_path: PDF 檔案路徑
        child_name: 兒童姓名
    
    Returns:
        (成功與否, 錯誤訊息)
    """
    try:
        # 生成 doc_id
        doc_id = f"report_{report_id}_{child_name.replace(' ', '_')}"
        
        # 調用 pdf_parser/pdf_processor_main.py 處理 PDF 並插入 Neo4j
        neo4j_config = {
            'uri': get_neo4j_uri(),
            'user': get_neo4j_auth()[0],
            'password': get_neo4j_auth()[1]
        }
        
        print(f"[INFO] 開始處理報告 {report_id}，使用 pdf_parser 模組，doc_id: {doc_id}")
        
        # 初始化 pipeline
        # archive_dir 可以根據需要更改，這裡預設為 uploads 目錄旁邊的 json_archives
        archive_dir = os.path.join(os.path.dirname(file_path), "json_archives")
        os.makedirs(archive_dir, exist_ok=True)
        
        pipeline = IEPPipeline(neo4j_config=neo4j_config, archive_dir=archive_dir)
        
        # 執行 pipeline (PDF -> JSON -> Neo4j)
        success = pipeline.run(file_path, child_id=doc_id)
        
        if not success:
            return False, "PDF 處理失敗：IEP Pipeline 執行出錯"
        
        # 獲取生成的 JSON 路徑（與 pipeline 中的邏輯一致）
        # 注意：pipeline.run 內部會生成帶時間戳的檔名，我們這裡嘗試預測或從 archive_dir 找最新的
        # 或者我們可以假設處理已成功，並依賴資料庫記錄。
        
        # 保存處理結果到資料庫
        # 1. 保存 KnowledgeGraphProcessing 記錄
        kg_processing = KnowledgeGraphProcessing(
            report_id=report_id,
            doc_id=doc_id,
            total_chunks=0, # pdf_parser 不直接返回這個計數，設為 0
            total_facets=0,
            total_relationships=0,
            neo4j_uri=neo4j_config['uri'],
            processing_time=0, # pipeline 不返回時間，設為 0
            status='success',
            error_message=None
        )
        db.session.add(kg_processing)
        
        db.session.commit()
        
        print(f"[INFO] 報告 {report_id} 處理完成")
        return True, "處理成功！已建立知識圖譜"
        
    except Exception as e:
        db.session.rollback()
        error_msg = str(e)
        print(f"[ERROR] 處理報告 {report_id} 失敗: {error_msg}")
        
        # 保存錯誤記錄
        try:
            # 使用預設值
            error_doc_id = f"report_{report_id}"
            error_neo4j_uri = get_neo4j_uri()
            
            kg_processing = KnowledgeGraphProcessing(
                report_id=report_id,
                doc_id=error_doc_id,
                total_chunks=0,
                total_facets=0,
                total_relationships=0,
                neo4j_uri=error_neo4j_uri,
                processing_time=0,
                status='failed',
                error_message=error_msg
            )
            db.session.add(kg_processing)
            db.session.commit()
        except Exception as save_error:
            print(f"[ERROR] 保存錯誤記錄失敗: {save_error}")
        
        return False, f"處理失敗：{error_msg}"

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        role = request.form['role']
        if User.query.filter_by(username=username).first():
            flash('用戶名已存在')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('電子郵件已存在')
            return render_template('register.html')
        user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role
        )
        db.session.add(user)
        db.session.commit()
        flash('註冊成功！請登入')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('用戶名或密碼錯誤')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'therapist':
        caregivers = User.query.filter_by(role='caregiver').all()
        children = Child.query.all()
        return render_template('therapist_dashboard.html', caregivers=caregivers, children=children)
    else:
        my_children = Child.query.filter_by(caregiver_id=current_user.id).all()
        return render_template('caregiver_dashboard.html', children=my_children, now=datetime.now().date())

@app.route('/add_caregiver', methods=['POST'])
@login_required
def add_caregiver():
    if current_user.role != 'therapist':
        flash('權限不足')
        return redirect(url_for('dashboard'))
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']
    if User.query.filter_by(username=username).first():
        flash('用戶名已存在')
        return redirect(url_for('dashboard'))
    if User.query.filter_by(email=email).first():
        flash('電子郵件已存在')
        return redirect(url_for('dashboard'))
    caregiver = User(username=username, email=email, password_hash=generate_password_hash(password), role='caregiver')
    db.session.add(caregiver)
    db.session.commit()
    flash('照護者帳號創建成功')
    return redirect(url_for('dashboard'))

@app.route('/add_child', methods=['POST'])
@login_required
def add_child():
    if current_user.role != 'therapist':
        flash('權限不足')
        return redirect(url_for('dashboard'))
    name = request.form['name']
    birth_date = datetime.strptime(request.form['birth_date'], '%Y-%m-%d').date()
    gender = request.form['gender']
    caregiver_id = request.form['caregiver_id']
    child = Child(name=name, birth_date=birth_date, gender=gender, caregiver_id=caregiver_id, therapist_id=current_user.id)
    db.session.add(child)
    db.session.commit()
    flash('兒童資料新增成功')
    return redirect(url_for('dashboard'))

@app.route('/upload_report', methods=['POST'])
@login_required
def upload_report():
    if current_user.role != 'therapist':
        flash('權限不足')
        return redirect(url_for('dashboard'))
    if 'file' not in request.files:
        flash('沒有選擇檔案')
        return redirect(url_for('dashboard'))
    file = request.files['file']
    child_id = request.form['child_id']
    if file.filename == '':
        flash('沒有選擇檔案')
        return redirect(url_for('dashboard'))
    if file and file.filename.lower().endswith('.pdf'):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 獲取兒童資訊
        child = db.session.get(Child, child_id)
        if not child:
            flash('找不到指定的兒童資料')
            return redirect(url_for('dashboard'))
        
        # 創建報告記錄
        report = Report(filename=filename, original_filename=file.filename, file_path=file_path, child_id=child_id)
        db.session.add(report)
        db.session.commit()
        
        # 上傳後自動處理 PDF（使用 process_pdf_to_graph 方式）
        try:
            success, message = process_report_after_upload(report.id, file_path, child.name)
            if success:
                flash(f'報告上傳成功！{message}')
            else:
                flash(f'報告上傳成功，但處理時發生錯誤：{message}')
        except Exception as e:
            flash(f'報告上傳成功，但處理時發生錯誤：{str(e)}')
    else:
        flash('請上傳PDF檔案')
    return redirect(url_for('dashboard'))

@app.route('/download_report/<int:report_id>')
@login_required
def download_report(report_id):
    report = db.session.get(Report, report_id)
    if not report:
        abort(404)
    if current_user.role == 'caregiver':
        if report.child.caregiver_id != current_user.id:
            flash('權限不足')
            return redirect(url_for('dashboard'))
    return send_file(report.file_path, as_attachment=True, download_name=report.original_filename)

@app.route('/view_report/<int:report_id>')
@login_required
def view_report(report_id):
    report = db.session.get(Report, report_id)
    if not report:
        abort(404)
    if current_user.role == 'caregiver' and report.child.caregiver_id != current_user.id:
        flash('權限不足')
        return redirect(url_for('dashboard'))
    return send_file(report.file_path)

# ============================================================================
# 對話與檢索相關助手函數
# ============================================================================

def get_doc_id_from_child(child_id: int) -> str:
    """
    從兒童 ID 獲取對應的 doc_id（從最新的報告中獲取）
    
    Args:
        child_id: 兒童 ID
    
    Returns:
        doc_id 或 None
    """
    try:
        child = db.session.get(Child, child_id)
        if not child or not child.reports:
            return None
        
        # 獲取最新的報告
        latest_report = max(child.reports, key=lambda r: r.uploaded_at)
        
        # 獲取對應的 KnowledgeGraphProcessing
        kg_processing = KnowledgeGraphProcessing.query.filter_by(
            report_id=latest_report.id,
            status='success'
        ).first()
        
        if kg_processing:
            return kg_processing.doc_id
        
        return None
    except Exception as e:
        print(f"[ERROR] 獲取 doc_id 失敗: {e}")
        return None

# 全域 LLM 生成器（延遲初始化）
_llm_generator = None

def get_llm_generator():
    """獲取 LLM 生成器（單例）"""
    global _llm_generator
    if _llm_generator is None:
        llm_config = LLMConfig()
        _llm_generator = LLMGenerator(config=llm_config)
        print("[LLM] LLM 生成器已初始化")
    return _llm_generator

def add_citation_boxes(response: str, candidates: list) -> str:
    """
    在回應尾端添加引用標註（只顯示頁數，去重）
    
    Args:
        response: LLM 生成的回應
        candidates: 檢索到的候選內容列表
    
    Returns:
        添加了引用標註的回應
    """
    if not candidates:
        return response
    
    # 提取頁數並去重
    seen_pages = set()
    pages = []
    
    for candidate in candidates:
        path = candidate.get('path', {})
        if not path:
            continue
        
        # 提取頁數
        page_start = path.get('page_start')
        page_end = path.get('page_end')
        
        # 格式化頁數
        if page_start is not None and page_end is not None:
            if page_start == page_end:
                page_str = f"第 {page_start} 頁"
                page_key = (page_start, page_start)
            else:
                page_str = f"第 {page_start}-{page_end} 頁"
                page_key = (page_start, page_end)
        elif page_start is not None:
            page_str = f"第 {page_start} 頁"
            page_key = (page_start, page_start)
        else:
            continue  # 沒有頁數就跳過
        
        # 檢查是否重複
        if page_key in seen_pages:
            continue
        seen_pages.add(page_key)
        
        pages.append((page_start, page_str))
    
    # 如果有引用，在回應尾端添加標註
    if pages:
        # 按頁數排序
        pages.sort(key=lambda x: x[0])
        
        # 提取頁數字串
        page_strings = [page_str for _, page_str in pages]
        
        citation_boxes = "\n\n【引用來源】\n"
        citation_boxes += "、".join(page_strings)
        response += citation_boxes
    
    return response


def generate_llm_response(
    user_query: str,
    retrieved_context: list,
    conversation_history: list = None,
    benchmark_mode: bool = False,
    # DST 相關參數（用於動態配置）
    semantic_flow: str = "continue",
    retrieval_action: str = "CONTEXT_FIRST",
    task_label: str = None,
    scope_label: str = None,
    is_ambiguous: bool = False,
    is_overview_query: bool = False,
    is_multi_domain: bool = False,
    top_domain: str = "",
    active_domains: list = None,
    domain_distribution: dict = None
) -> str:
    """
    使用 LLM 生成回應

    Args:
        user_query: 使用者查詢
        retrieved_context: 檢索到的上下文資料
        conversation_history: 對話歷史（可選）
        benchmark_mode: 基準測試模式（不使用 retrieved_context）
        semantic_flow: 語義流程類型
        retrieval_action: 檢索動作
        task_label: Task 類型
        scope_label: Scope 類型
        is_ambiguous: 是否模糊
        is_overview_query: 是否為整體查詢
        is_multi_domain: 是否多領域
        top_domain: 頂級領域

    Returns:
        LLM 生成的回應
    """
    llm_generator = get_llm_generator()
    
    # 如果 benchmark_mode，不使用檢索上下文
    context_to_use = [] if benchmark_mode else (retrieved_context or [])
    
    # 獲取動態配置
    prompt_manager = LLMPromptManager()
    generation_config = prompt_manager.get_config(
        semantic_flow=semantic_flow,
        retrieval_action=retrieval_action,
        task_label=task_label,
        scope_label=scope_label,
        is_ambiguous=is_ambiguous,
        is_overview_query=is_overview_query,
        is_multi_domain=is_multi_domain,
        top_domain=top_domain,
        active_domains=active_domains or [],
        domain_distribution=domain_distribution or {}
    )
    
    # 將模糊相關資訊添加到 config 中（用於 build_user_prompt）
    generation_config.is_ambiguous = is_ambiguous
    generation_config.active_domains = active_domains or []
    
    print(f"[LLM] 使用配置：temperature={generation_config.temperature:.2f}, "
          f"max_tokens={generation_config.max_tokens}, "
          f"response_style={generation_config.response_style}, "
          f"context_format={generation_config.context_format_style}")
    
    return llm_generator.generate_response(
        user_query=user_query,
        retrieved_context=context_to_use,
        conversation_history=conversation_history,
        generation_config=generation_config
    )

# 全域檢索器（延遲初始化）
_graph_client = None
_retrieval_v2 = None
_bge_model = None

def get_retrieval_components():
    """獲取檢索組件（單例）"""
    global _graph_client, _retrieval_v2, _bge_model
    if _graph_client is None:
        _graph_client = GraphClient(
            uri=app.config['NEO4J_URI'],
            user=app.config['NEO4J_USER'],
            password=app.config['NEO4J_PASSWORD']
        )
        
        # 確保 DST 組件已初始化以獲得 TextEncoder
        try:
            init_dst_components()
            if _shared_encoder is not None:
                _bge_model = _shared_encoder
                print("[RetrievalV2] 使用共用的 TextEncoder (遠端 API)")
            else:
                _bge_model = None
        except Exception as e:
            print(f"[RetrievalV2] BGE-M3 模型載入失敗: {e}")
            _bge_model = None
            
        # 初始化新的檢索模組 V2
        _retrieval_v2 = RetrievalModuleV2(_graph_client, sql_db=db, text_encoder=_bge_model)
        print("[RetrievalV2] 檢索模組 V2 已初始化")
        
    return _graph_client, _retrieval_v2

@app.route('/api/chat', methods=['POST'])
@login_required
def chat(): # 聊天 API - 處理用戶消息並進行對話狀態追蹤
    """
    聊天 API - 處理用戶消息並進行對話狀態追蹤
    
    請求格式:
    {
        "message": "使用者訊息",
        "child_id": 兒童ID（可選）
    }
    
    回應格式:
    {
        "response": "助理回應",
        "flow_state": {...},
        "retrieval_info": {...}
    }
    """
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        child_id = data.get('child_id')
        
        if not user_message:
            return jsonify({"error": "消息不能為空"}), 400
        
        if not child_id:
            return jsonify({"error": "必須指定兒童 ID"}), 400
        
        # 驗證兒童存在且屬於當前用戶
        child = db.session.get(Child, child_id)
        if not child or (child.caregiver_id != current_user.id and child.therapist_id != current_user.id):
            return jsonify({"error": "無權訪問此兒童"}), 403
        
        # 獲取或創建此用戶/兒童的對話狀態分類器
        classifier = get_dialogue_classifier(current_user.id, child_id)
        
        # 運行 DST 分析（先不傳 assistant_reply，因為還沒生成）
        # 注意：predict() 內部會更新狀態（用戶輸入）和 turn_index
        flow_result = classifier.predict(user_message)
        
        # 印出對話狀態模組的完整分析結果
        print("\n" + "=" * 80)
        print(f"[對話狀態追蹤] Turn {flow_result.turn_index}")
        print("=" * 80)
        print(f"用戶訊息: {user_message}")
        print("-" * 80)
        
        # 1. 領域分析 (Domain Analysis)
        print("\n【1. 領域分析 (Domain Analysis)】")
        print(f"  頂級領域: {flow_result.domain_analysis.top_domain}")
        print(f"  頂級機率: {flow_result.domain_analysis.top_prob:.4f}")
        print(f"  正規化熵: {flow_result.domain_analysis.entropy:.4f} (0=確定, 1=不確定)")
        print(f"  是否多領域: {flow_result.domain_analysis.is_multi_domain}")
        print(f"  活躍領域數量: {len(flow_result.domain_analysis.active_domains)}")
        print(f"  活躍領域列表: {flow_result.domain_analysis.active_domains}")
        if flow_result.domain_analysis.active_domain_probs:
            print(f"  活躍領域機率分布:")
            for domain, prob in sorted(flow_result.domain_analysis.active_domain_probs.items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"    - {domain}: {prob:.4f}")
        print(f"  完整領域分布 (Top 5):")
        sorted_domains = sorted(flow_result.domain_analysis.distribution.items(), 
                               key=lambda x: x[1], reverse=True)[:5]
        for domain, prob in sorted_domains:
            print(f"    - {domain}: {prob:.4f}")
        
        # 2. 上下文相似度分析 (Context Similarity)
        print("\n【2. 上下文相似度分析 (Context Similarity)】")
        print(f"  相似度分數 (C): {flow_result.context_analysis.similarity_score:.4f}")
        print(f"  來源: {flow_result.context_analysis.source}")
        print(f"  是否第一輪: {flow_result.context_analysis.is_first_turn}")
        if flow_result.context_analysis.source == "prev_user":
            print(f"  → 與上一輪用戶輸入相似")
        elif flow_result.context_analysis.source == "prev_bot":
            print(f"  → 與上一輪機器人回覆相似")
        elif flow_result.context_analysis.source == "first_turn":
            print(f"  → 第一輪對話，無歷史上下文")
        
        # 3. 主題延續分析 (Topic Continuation)
        print("\n【3. 主題延續分析 (Topic Continuation)】")
        print(f"  是否延續: {flow_result.topic_analysis.is_continuing}")
        print(f"  主題重疊分數 (MT): {flow_result.topic_analysis.overlap_score:.4f}")
        print(f"  判斷原因: {flow_result.topic_analysis.reason}")
        is_ambiguous_flow = flow_result.policy_decision.is_ambiguous
        if flow_result.topic_analysis.tv_distance is not None:
            print(f"  TV 距離: {flow_result.topic_analysis.tv_distance:.4f} (0=相同, 1=完全不同)")
            # 非模糊情境：TV 可作為記憶重置訊號
            if not is_ambiguous_flow:
                if flow_result.topic_analysis.tv_distance >= 0.6:
                    print(f"    → TV >= 0.6，判定為強切換，記憶已重置")
                else:
                    print(f"    → TV < 0.6，正常更新記憶（EMA）")
            else:
                # 高熵「模糊」情境：TV 僅供觀察，是否延續/回退交由 DST 模糊規則決定
                print(f"    → 高熵模糊狀態，TV 僅供觀察，不作為記憶重置依據")
        if flow_result.topic_analysis.prev_top_domain:
            print(f"  上一輪頂級領域: {flow_result.topic_analysis.prev_top_domain}")
        if flow_result.topic_analysis.cur_top_domain:
            print(f"  當前頂級領域: {flow_result.topic_analysis.cur_top_domain}")
        
        # 4. 策略決策 (Policy Decision)
        print("\n【4. 策略決策 (Policy Decision)】")
        print(f"  語義流程類型: {flow_result.policy_decision.semantic_flow.upper()}")
        print(f"     → continue: 強延續 | shift_soft: 弱切換 | shift_hard: 硬切換")
        print(f"  檢索動作: {flow_result.policy_decision.retrieval_action}")
        print(f"     → NARROW_GRAPH: 縮小檢索範圍")
        print(f"     → CONTEXT_FIRST: 優先使用上下文")
        print(f"     → WIDE_IN_DOMAIN: 領域內廣泛檢索")
        print(f"     → DUAL_OR_CLARIFY: 雙路檢索或澄清")
        print(f"  上下文級別: {flow_result.policy_decision.context_level.upper()}")
        print(f"  是否模糊: {flow_result.policy_decision.is_ambiguous}")
        print(f"  策略案例代碼: {flow_result.policy_decision.policy_case}")
        print(f"     → 格式: C[L/H]_MT[L/H]_[ACTION]_[MODIFIERS]")
        print(f"     → C: Context, MT: Multi-Topic")
        print(f"     → L: Low, H: High")
        print(f"     → MODIFIERS: AMBIG(模糊), MD(多領域)")
        
        # 5. 任務/範圍分類 (Task/Scope Classification)
        print("\n【5. 任務/範圍分類 (Task/Scope Classification)】")
        if flow_result.task_label:
            print(f"  任務類型: {flow_result.task_label}")
            if flow_result.task_dist:
                print(f"  任務分布 (Top 3):")
                sorted_tasks = sorted(flow_result.task_dist.items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
                for task, prob in sorted(sorted_tasks):
                    print(f"    - {task}: {prob:.4f}")
        else:
            print(f"  任務類型: 未分類")
        if flow_result.scope_label:
            print(f"  範圍類型: {flow_result.scope_label}")
            if flow_result.scope_dist:
                print(f"  範圍分布 (Top 3):")
                sorted_scopes = sorted(flow_result.scope_dist.items(), 
                                      key=lambda x: x[1], reverse=True)[:3]
                for scope, prob in sorted_scopes:
                    print(f"    - {scope}: {prob:.4f}")
        else:
            print(f"  範圍類型: 未分類")
        
        # 6. 決策摘要
        print("\n【決策摘要】")
        print(f"  基於 (C={flow_result.context_analysis.similarity_score:.3f}, "
              f"MT={flow_result.topic_analysis.overlap_score:.3f}, "
              f"entropy={flow_result.domain_analysis.entropy:.3f})")
        if flow_result.topic_analysis.tv_distance is not None:
            print(f"  TV 距離: {flow_result.topic_analysis.tv_distance:.3f}")
        if flow_result.task_label:
            print(f"  任務: {flow_result.task_label}")
        print(f"  → 判斷為: {flow_result.policy_decision.semantic_flow.upper()}")
        print(f"  → 建議檢索策略: {flow_result.policy_decision.retrieval_action}")
        
        print("\n" + "=" * 80 + "\n")
        
        # 構建 turn_state（用於檢索規劃器）
        # 優先順序：整體分布 > 融合後的分布 > 原始分布
        if flow_result.domain_analysis.is_overview_query and flow_result.domain_analysis.overview_distribution:
            domain_distribution = flow_result.domain_analysis.overview_distribution
            print(f"[檢索規劃器] 使用整體查詢分布")
        elif flow_result.domain_analysis.fused_distribution is not None:
            domain_distribution = flow_result.domain_analysis.fused_distribution
            print(f"[檢索規劃器] 使用融合後的分布")
        else:
            domain_distribution = flow_result.domain_analysis.distribution
            print(f"[檢索規劃器] 使用原始分布")
        
        turn_state = {
            "retrieval_action": flow_result.policy_decision.retrieval_action,
            "domain_distribution": domain_distribution,  # 使用融合後的分布（如果存在）
            "task_dist": flow_result.task_dist or {},
            "task_pred": flow_result.task_label or "",
            "scope_pred": flow_result.scope_label or "",
            "semantic_flow": flow_result.policy_decision.semantic_flow,
            "top_domain": flow_result.domain_analysis.top_domain,
            "top_domain_prob": flow_result.domain_analysis.top_prob,
            "topic_overlap": flow_result.topic_analysis.overlap_score,
            "turn_index": flow_result.turn_index,
            "is_ambiguous": flow_result.policy_decision.is_ambiguous,
            "normalized_entropy": flow_result.domain_analysis.entropy,
            # DST 提供的領域範圍信息
            "active_domains": flow_result.domain_analysis.active_domains,  # 本輪的活躍領域
            "prev_active_domains": flow_result.topic_analysis.prev_active_domains or [],  # 上一輪的活躍領域
            "fused_distribution_used": flow_result.domain_analysis.fused_distribution is not None,  # DST 是否觸發了模糊延續
            "detected_region": flow_result.detected_region,  # 偵測到的地區
        }
        
        # 執行檢索 (使用 RetrievalModuleV2)
        try:
            graph_client, retrieval_v2 = get_retrieval_components()
            
            # 獲取 doc_id（從最新的報告中獲取）
            doc_id = get_doc_id_from_child(child_id)
            if not doc_id:
                raise ValueError(f"找不到兒童 {child_id} 的有效報告 doc_id，請先上傳報告")
            
            # 直接調用 V2 檢索入口
            # retrieve 內部會自動處理 Strategy Mapping -> Execution -> Reranking
            candidates = retrieval_v2.retrieve(
                turn_state=turn_state,
                user_query=user_message,
                doc_id=str(doc_id)
            )
            
            print("\n" + "=" * 80)
            print(f"[RetrievalModuleV2] 檢索完成，找到 {len(candidates)} 個候選節點")
            print("=" * 80)
            
            if candidates:
                print("\n【全部檢索到的內容 (按分數排序)】")
                for i, c in enumerate(candidates, 1):
                    # 印出完整或較長內容以便除錯
                    print(f"  {i}. [Score: {c.score:.4f}] [{c.label}]")
                    # 將內容縮進並印出
                    indented_text = c.text.replace("\n", "\n      ")
                    print(f"      {indented_text}")
                    print("-" * 40)
            
            print("=" * 80 + "\n")
            
            # 轉換為 LLM 可用的 context 格式 (保持向後兼容)
            retrieved_context = []
            for c in candidates:
                retrieved_context.append({
                    "text": c.text,
                    "score": c.score,
                    "label": c.label,
                    "id": c.node_id,
                    "path": {
                        "subdomain": c.properties.get("subdomain", ""),
                        "category": c.properties.get("category", "")
                    }
                })
            
            print("=" * 80 + "\n")
            
            # 構建對話歷史（從資料庫獲取最近的對話）
            # 判斷是否需要跳過對話歷史：
            # 1. 記憶重置：
            #    - 非模糊情況下：TV 距離 >= 0.6 或 reason 包含 "mem_reset"
            #    - 無論是否模糊：semantic_flow == "shift_hard"
            # 2. 主題延續轉換：從 continue → shift_soft/shift_hard 或 shift_soft → shift_hard
            should_skip_history = False
            skip_reason = ""
            
            # 檢查記憶重置
            is_memory_reset = False
            is_ambiguous_flow = flow_result.policy_decision.is_ambiguous

            # TV / mem_reset 僅在「非模糊」情況下視為記憶重置訊號
            if (
                not is_ambiguous_flow
                and flow_result.topic_analysis.tv_distance is not None
                and flow_result.topic_analysis.tv_distance >= 0.6
            ):
                is_memory_reset = True
                skip_reason = (
                    f"記憶重置（TV 距離={flow_result.topic_analysis.tv_distance:.4f} >= 0.6）"
                )
            elif flow_result.policy_decision.semantic_flow == "shift_hard":
                is_memory_reset = True
                skip_reason = "記憶重置（semantic_flow=shift_hard）"
            elif (
                not is_ambiguous_flow
                and "mem_reset" in (flow_result.topic_analysis.reason or "")
            ):
                is_memory_reset = True
                skip_reason = (
                    f"記憶重置（reason 包含 mem_reset: {flow_result.topic_analysis.reason}）"
                )
            
            # 檢查主題延續轉換（需要讀取上一輪的 semantic_flow）
            is_topic_continuation_shift = False
            if not is_memory_reset and flow_result.turn_index > 0:
                try:
                    # 從對話緩存讀取上一輪的 semantic_flow
                    cache_file = os.path.join(DIALOGUE_CACHE_DIR, f"user_{current_user.id}_child_{child_id}_dialogue.json")
                    if os.path.exists(cache_file):
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            cache_data = json.load(f)
                        
                        # 找到上一輪的記錄
                        prev_turn_index = flow_result.turn_index - 1
                        prev_semantic_flow = None
                        for entry in reversed(cache_data):
                            if entry.get("turn_index") == prev_turn_index:
                                prev_semantic_flow = entry.get("dst_analysis", {}).get("policy_decision", {}).get("semantic_flow")
                                break
                        
                        if prev_semantic_flow:
                            current_semantic_flow = flow_result.policy_decision.semantic_flow
                            # 檢查是否發生轉換：continue → shift_soft/shift_hard 或 shift_soft → shift_hard
                            if (prev_semantic_flow == "continue" and current_semantic_flow in ["shift_soft", "shift_hard"]) or \
                               (prev_semantic_flow == "shift_soft" and current_semantic_flow == "shift_hard"):
                                is_topic_continuation_shift = True
                                skip_reason = f"主題延續轉換（{prev_semantic_flow} → {current_semantic_flow}）"
                except Exception as e:
                    print(f"[WARNING] 讀取上一輪語義流程失敗: {e}")
            
            should_skip_history = is_memory_reset or is_topic_continuation_shift
            
            conversation_history = []
            if not should_skip_history:
                try:
                    recent_messages = ChatMessage.query.filter_by(
                        user_id=current_user.id,
                        child_id=child_id
                    ).order_by(ChatMessage.sent_at.desc()).limit(10).all()
                    
                    # 反轉順序（從舊到新）
                    for msg in reversed(recent_messages):
                        role = "user" if msg.is_user_message else "assistant"
                        conversation_history.append({
                            "role": role,
                            "content": msg.message
                        })
                except Exception as e:
                    print(f"[WARNING] 獲取對話歷史失敗: {e}")
            else:
                print(f"[對話歷史] 跳過對話歷史：{skip_reason}")
            
            # 使用 LLM 生成回應（使用檢索結果和 DST 配置）
            print("[LLM] 開始生成回應...")
            bot_response = generate_llm_response(
                user_query=user_message,
                retrieved_context=retrieved_context,
                conversation_history=conversation_history,
                semantic_flow=flow_result.policy_decision.semantic_flow,
                retrieval_action=flow_result.policy_decision.retrieval_action,
                task_label=flow_result.task_label,
                scope_label=flow_result.scope_label,
                is_ambiguous=flow_result.policy_decision.is_ambiguous,
                is_overview_query=flow_result.domain_analysis.is_overview_query,
                is_multi_domain=flow_result.domain_analysis.is_multi_domain,
                top_domain=flow_result.domain_analysis.top_domain,
                active_domains=flow_result.domain_analysis.active_domains,
                domain_distribution=domain_distribution
            )
            print(f"[LLM] 回應生成完成（長度：{len(bot_response)} 字元）")
            
            # 在回應尾端添加引用標註
            bot_response = add_citation_boxes(bot_response, retrieved_context)
            
        except Exception as e:
            print(f"[檢索錯誤] {e}")
            traceback.print_exc()
            # Fallback: 使用 LLM 生成回應（沒有檢索上下文）
            try:
                print("[LLM] 檢索失敗，使用 LLM 生成基本回應...")
                bot_response = generate_llm_response(
                    user_query=user_message,
                    retrieved_context=[],
                    conversation_history=None,
                    semantic_flow=flow_result.policy_decision.semantic_flow,
                    retrieval_action=flow_result.policy_decision.retrieval_action,
                    task_label=flow_result.task_label,
                    scope_label=flow_result.scope_label,
                    is_ambiguous=flow_result.policy_decision.is_ambiguous,
                    is_overview_query=flow_result.domain_analysis.is_overview_query,
                    is_multi_domain=flow_result.domain_analysis.is_multi_domain,
                    top_domain=flow_result.domain_analysis.top_domain,
                    active_domains=flow_result.domain_analysis.active_domains,
                    domain_distribution=flow_result.domain_analysis.distribution
                )
                # Fallback 情況下沒有檢索結果，不需要添加引用標註
            except Exception as llm_error:
                print(f"[LLM] LLM 生成失敗: {llm_error}")
                # 最後的 fallback
                bot_response = f"抱歉，我目前無法回答關於 {flow_result.domain_analysis.top_domain} 的問題。請稍後再試。"
        
        # 更新機器人回覆到狀態（用戶輸入已在 predict() 中更新，避免重複編碼）
        classifier.context_similarity.update_bot_only(bot_response)
        
        # 保存對話狀態到文件（持久化）
        classifier.save_state(current_user.id, child_id)
        
        # 保存用戶消息到數據庫
        user_chat = ChatMessage(
            user_id=current_user.id,
            child_id=child_id,
            message=user_message,
            is_user_message=True,
            flow_state=json.dumps({
                "domain": flow_result.domain_analysis.top_domain,
                "normalized_entropy": float(flow_result.domain_analysis.entropy),
                "context_similarity": float(flow_result.context_analysis.similarity_score),
                "is_continuing": flow_result.topic_analysis.is_continuing,
                "semantic_flow": flow_result.policy_decision.semantic_flow,
                "retrieval_action": flow_result.policy_decision.retrieval_action
            }),
            retrieval_info=json.dumps({
                "flow_type": flow_result.policy_decision.semantic_flow,
                "retrieval_action": flow_result.policy_decision.retrieval_action,
                "context_level": flow_result.policy_decision.context_level
            })
        )
        db.session.add(user_chat)
        
        # 保存機器人回應
        bot_chat = ChatMessage(
            user_id=current_user.id,
            child_id=child_id,
            message=bot_response,
            is_user_message=False,
            flow_state=None,
            retrieval_info=None
        )
        db.session.add(bot_chat)
        db.session.commit()
        
        # 保存到對話緩存（包含完整分析結果）
        try:
            cache_file = os.path.join(DIALOGUE_CACHE_DIR, f"user_{current_user.id}_child_{child_id}_dialogue.json")
            if not os.path.exists(cache_file):
                cache_data = []
            else:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
            
            # 構建完整的對話記錄（包含所有分析結果）
            dialogue_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "turn_index": flow_result.turn_index,
                "user_message": user_message,
                "bot_response": bot_response,
                # 完整的 DST 分析結果
                "dst_analysis": flow_result.to_dict()
            }
            
            cache_data.append(dialogue_entry)
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            print(f"[CACHE] 已保存完整分析結果到對話緩存 (Turn {flow_result.turn_index})")
        except Exception as e:
            print(f"[CACHE] 保存對話緩存失敗: {e}")
            traceback.print_exc()
        
        return jsonify({
            "response": bot_response,
            "flow_state": {
                "domain": flow_result.domain_analysis.top_domain,
                "normalized_entropy": float(flow_result.domain_analysis.entropy),
                "semantic_flow": flow_result.policy_decision.semantic_flow,
                "context_similarity": float(flow_result.context_analysis.similarity_score)
            },
            "retrieval_info": {
                "retrieval_action": flow_result.policy_decision.retrieval_action
            }
        }), 200
    
    except Exception as e:
        print(f"[ERROR] /api/chat 錯誤: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/reset', methods=['POST'])
@login_required
def reset_chat():
    """
    重置對話記憶和對話紀錄
    
    請求格式:
    {
        "child_id": 兒童ID（必需）
    }
    
    回應格式:
    {
        "success": true/false,
        "message": "重置成功" 或錯誤訊息
    }
    """
    try:
        data = request.get_json()
        child_id = data.get('child_id')
        
        if not child_id:
            return jsonify({"error": "必須指定兒童 ID"}), 400
        
        # 驗證兒童存在且屬於當前用戶
        child = db.session.get(Child, child_id)
        if not child or (child.caregiver_id != current_user.id and child.therapist_id != current_user.id):
            return jsonify({"error": "無權訪問此兒童"}), 403
        
        # 1. 重置 DST 分類器
        global _user_classifiers
        key = (current_user.id, child_id)
        if key in _user_classifiers:
            classifier = _user_classifiers[key]
            classifier.reset()
            print(f"[RESET] 已重置用戶 {current_user.id} 的兒童 {child_id} 的 DST 分類器")
        
        # 2. 刪除對話狀態文件
        try:
            from dialogue_state_module.state_persistence import delete_dialogue_state
            delete_dialogue_state(current_user.id, child_id)
            print(f"[RESET] 已刪除用戶 {current_user.id} 的兒童 {child_id} 的對話狀態文件")
        except Exception as e:
            print(f"[RESET] 刪除對話狀態文件失敗: {e}")
        
        # 3. 刪除資料庫中的對話紀錄 (用戶要求保留紀錄，不再刪除 MySQL 資料)
        # try:
        #     deleted_count = ChatMessage.query.filter_by(
        #         user_id=current_user.id,
        #         child_id=child_id
        #     ).delete()
        #     db.session.commit()
        #     print(f"[RESET] 已保留 {deleted_count} 條對話紀錄 (不從資料庫刪除)")
        # except Exception as e:
        #     db.session.rollback()
        #     print(f"[RESET] 操作資料庫失敗: {e}")
        
        # 4. 刪除對話緩存文件
        try:
            cache_file = os.path.join(DIALOGUE_CACHE_DIR, f"user_{current_user.id}_child_{child_id}_dialogue.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"[RESET] 已刪除對話緩存文件: {cache_file}")
        except Exception as e:
            print(f"[RESET] 刪除對話緩存文件失敗: {e}")
        
        # 5. 重置檢索狀態
        try:
            # 已移除舊版 RetrievalState 引用
            pass
        except Exception as e:
            print(f"[RESET] 重置檢索狀態失敗: {e}")
        
        return jsonify({
            "success": True,
            "message": "已成功重置對話記憶和對話紀錄"
        }), 200
    
    except Exception as e:
        print(f"[ERROR] /api/chat/reset 錯誤: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/chat/history', methods=['GET'])
@login_required
def get_chat_history():
    """
    獲取對話歷史
    
    Query Parameters:
        child_id: 兒童 ID（必需）
    
    回應格式:
    {
        "messages": [
            {
                "id": 消息 ID,
                "message": "消息內容",
                "is_user_message": true/false,
                "sent_at": "時間戳",
                "flow_state": {...},
                "retrieval_info": {...}
            },
            ...
        ]
    }
    """
    try:
        child_id = request.args.get('child_id')
        if not child_id:
            return jsonify({"error": "必須指定兒童 ID"}), 400
        
        # 驗證兒童存在且屬於當前用戶
        child = db.session.get(Child, child_id)
        if not child or (child.caregiver_id != current_user.id and child.therapist_id != current_user.id):
            return jsonify({"error": "無權訪問此兒童"}), 403
        
        # 從對話緩存 JSON 讀取歷史
        cache_file = os.path.join(DIALOGUE_CACHE_DIR, f"user_{current_user.id}_child_{child_id}_dialogue.json")
        history = []
        
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                
                # 將緩存中的「回合（Turn）」平坦化為「消息（Message）」格式
                for i, entry in enumerate(cache_data):
                    sent_at = entry.get("timestamp")
                    dst = entry.get("dst_analysis", {})
                    
                    # 1. 加入用戶消息
                    user_msg = {
                        "id": i * 2,
                        "message": entry.get("user_message", ""),
                        "is_user_message": True,
                        "sent_at": sent_at,
                        "flow_state": {
                            "domain": dst.get("domain_analysis", {}).get("top_domain"),
                            "normalized_entropy": dst.get("domain_analysis", {}).get("entropy"),
                            "context_similarity": dst.get("context_analysis", {}).get("similarity_score"),
                            "is_continuing": dst.get("topic_analysis", {}).get("is_continuing"),
                            "semantic_flow": dst.get("policy_decision", {}).get("semantic_flow"),
                            "retrieval_action": dst.get("policy_decision", {}).get("retrieval_action")
                        },
                        "retrieval_info": {
                            "flow_type": dst.get("policy_decision", {}).get("semantic_flow"),
                            "retrieval_action": dst.get("policy_decision", {}).get("retrieval_action"),
                            "context_level": dst.get("policy_decision", {}).get("context_level")
                        }
                    }
                    history.append(user_msg)
                    
                    # 2. 加入機器人回應
                    bot_msg = {
                        "id": i * 2 + 1,
                        "message": entry.get("bot_response", ""),
                        "is_user_message": False,
                        "sent_at": sent_at, # 緩存中通常只有一個時間，這裡沿用
                        "flow_state": None,
                        "retrieval_info": None
                    }
                    history.append(bot_msg)
            except Exception as e:
                print(f"[ERROR] 讀取對話緩存失敗: {e}")
                # 讀取失敗則返回空列表，避免崩潰
        
        return jsonify({"messages": history}), 200
    
    except Exception as e:
        print(f"[ERROR] /api/chat/history 錯誤: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("=" * 60)
        print("早療系統啟動中...")
        print("=" * 60)
        try:
            # 初始化對話狀態追蹤模組（TextEncoder 會自動加載 BGE 模型）
            print("\n正在初始化對話狀態追蹤模組...")
            init_dst_components()
            print("=" * 60)
        except Exception as e:
            print(f"[FATAL] DST 初始化失敗: {e}")
            traceback.print_exc()
            print("=" * 60)
            raise  # Re-raise to stop the app
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)