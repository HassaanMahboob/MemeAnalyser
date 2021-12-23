import meme_identification as mi
import meme_classification as mc

from fastapi import Body
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


identifier_model = mi.init_model()
meme_db = mi.load_memes_db()

classifier_model, classifier_tokenizer = mc.init_model()



origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Hello World"}





@app.post("/is_meme/")
async def read_item(image: str = Body(...)):
    image = image[image.find("base64,")+7:]
    image = bytes(image , 'utf-8')
    import base64
    with open("test.png", "wb") as fh:
        fh.write(base64.decodebytes(image))
    result = mi.is_meme("test.png",meme_db, identifier_model)
    if result==1:
        return "It is a meme."
    else:
        return "It is not a meme."


@app.post("/is_hateful_meme/")
async def read_item(image: str = Body(...)):
    image = image[image.find("base64,")+7:]
    image = bytes(image , 'utf-8')
    import base64
    with open("test.png", "wb") as fh:
        fh.write(base64.decodebytes(image))
    result = mc.is_hateful("test.png", classifier_model, classifier_tokenizer)
    return f"Probability of hatefulness = {result}"
