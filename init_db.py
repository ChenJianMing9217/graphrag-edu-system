#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
資料庫初始化腳本（app_v4）
"""

from app import app, db, User
from werkzeug.security import generate_password_hash

def init_database():
    with app.app_context():
        db.create_all()
        print("✓ 資料庫表已創建")
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(
                username='admin',
                email='admin@example.com',
                password_hash=generate_password_hash('admin123'),
                role='therapist'
            )
            db.session.add(admin)
            db.session.commit()
            print("✓ 預設管理員帳號已創建 (用戶名: admin, 密碼: admin123)")
        else:
            print("✓ 管理員帳號已存在")
        print("✓ 資料庫初始化完成")

if __name__ == '__main__':
    init_database()

