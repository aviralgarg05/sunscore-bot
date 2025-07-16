import pandas as pd
from pymongo import MongoClient

MONGODB_URI="mongodb+srv://dataxheimat:RxKjoqvpz2yUoU92@data.fq8ofzs.mongodb.net/?retryWrites=true&w=majority&appName=Data"
MONGODB_DATABASE="SunScore"
MONGODB_COLLECTION="SolarData"
OUTPUT_FILE = "output.csv"

client = MongoClient(MONGODB_URI)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]
data = list(collection.find())

for doc in data:
    doc['_id'] = str(doc['_id'])

df = pd.DataFrame(data)
df.to_csv(OUTPUT_FILE, index=False)

print(f"Exported {len(df)} documents to {OUTPUT_FILE}")