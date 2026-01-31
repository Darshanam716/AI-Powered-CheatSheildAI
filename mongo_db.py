from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["cheatshieldai"]

students_col = db["students"]
attendance_col = db["attendance"]
exam_logs_col = db["exam_logs"]
violations_col = db["violations"]

print(" MongoDB connected successfully")
