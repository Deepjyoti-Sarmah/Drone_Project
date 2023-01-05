import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://faceattendencerealtime-50ecf-default-rtdb.firebaseio.com/"
})

ref = db.reference('Students')

data = {
    "01":
        {
            "name": "Linus",
            "works": "Linux",
            "staring_year": 2008,
            "total_attendence": 16,
            "standing": "g",
            "year": 4,
            "last_attendence_time": "2023-01-05 00:54:34"
        },
    "02":
        {
            "name": "Ana",
            "works": "Actress",
            "staring_year": 2010,
            "total_attendence": 10,
            "standing": "g",
            "year": 10,
            "last_attendence_time": "2023-01-05 00:54:34"
        },
    "03":
        {
            "name": "Elon",
            "works": "Tesla",
            "staring_year": 2008,
            "total_attendence": 5,
            "standing": "g",
            "year": 12,
            "last_attendence_time": "2023-01-05 00:54:34"
        },
    "04":
        {
            "name": "Deepjyoti",
            "works": "Drone",
            "staring_year": 2019,
            "total_attendence": 12,
            "standing": "g",
            "year": 3,
            "last_attendence_time": "2023-01-05 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)
