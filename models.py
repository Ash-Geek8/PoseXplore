from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class AngleData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    joint_name = db.Column(db.String(100))
    min_angle = db.Column(db.Float)
    max_angle = db.Column(db.Float)
    timestamp = db.Column(db.String(100))
