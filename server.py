import meme_identification as mi
import meme_classification as mc

import tagme
import json

from fastapi import Body
from fastapi import FastAPI

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

tagme.GCUBE_TOKEN = "dbc68d3b-0127-4709-9608-761d8e6c585f-843339462"
API_TOKEN = "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"

identifier_model = mi.init_model()
task, generator, models, use_cuda, use_fp16, pad_idx, bos_item, eos_item, patch_resize_transform = mc.init_caption_model()
pipeline = mc.init_ocr_model()
enginedbpedia = mc.init_engine()





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
    result = mi.predict_is_meme("test.png", identifier_model)
    return result


@app.post("/is_hateful_meme/")
async def read_item(image: str = Body(...)):
    image = image[image.find("base64,")+7:]
    image = bytes(image , 'utf-8')
    import base64
    with open("test.png", "wb") as fh:
        fh.write(base64.decodebytes(image))
    img_path = "test.png"
    meme_caption = mc.OCR_img2txt(img_path)
    image_caption = mc.get_caption(img_path, pipeline, task, generator, models, use_cuda, use_fp16, pad_idx, bos_item, eos_item, patch_resize_transform)
    context = ""
    prompt = f"""
    Classify the following data as hateful or not hateful:

    Meme Caption: 'Black culture is as shit as obama'
    Image Caption: A man in a suit and tie with his hand on his face
    Context: Black people is a racialized classification of people, usually a political and skin color-based category for specific populations with a mid to dark brown complexion. Not all people considered "black" have dark skin; in certain countries, often in socially based systems of racial classification in the Western world, the term "black" is used to describe persons who are perceived as dark-skinned compared to other populations.
    Shit is a word considered to be vulgar and profane in Modern English. As a noun, it refers to fecal matter, and as a verb it means to defecate; in the plural, it means diarrhoea.
    Barack Hussein Obama II is an American politician, lawyer, and author who served as the 44th president of the United States from 2009 to 2017. A member of the Democratic Party, Obama was the first African-American president of the United States.
    Result: Hateful
    ###
    Meme Caption: 'Silly women that's not kitchen'
    Image Caption: a woman wearing blue gloves working on a motorcycle
    Context: A kitchen is a room or part of a room used for cooking and food preparation in a dwelling or in a commercial establishment. A modern middle-class residential kitchen is typically equipped with a stove, a sink with hot and cold running water, a refrigerator, and worktops and kitchen cabinets arranged according to a modular design.
    A woman is an adult female human. Prior to adulthood, a female human is referred to as a girl (a female child or adolescent).
    Result: Hateful
    ###
    Meme Caption: 'when you ask ya parents a question and it turns into a lecture'
    Image Caption: a child in resentment with a pillow beside him
    Context: In mythology, folklore and speculative fiction, shapeshifting is the ability to physically transform oneself through an inherently superhuman ability, divine intervention, demonic manipulation, sorcery, spells or having inherited the ability. The idea of shapeshifting is in the oldest forms of totemism and shamanism, as well as the oldest existent literature and epic poems such as the Epic of Gilgamesh and the Iliad.
    Result: Not Hateful
    ###
    Meme Caption: 'your black neighbour after you called the cops'
    Image Caption: a chimpanzee with its mouth open
    Context: A neighbourhood (British English, Hibernian English, Australian English and Canadian English) or neighborhood (American English; see spelling differences) is a geographically localised community within a larger city, town, suburb or rural area. Neighbourhoods are often social communities with considerable face-to-face interaction among members.
    African Americans (also referred to as Black Americans and formerly, Afro-Americans) are an ethnic group consisting of Americans with partial or total ancestry from any of the Black racial groups of Africa. The term African American generally denotes descendants of enslaved Africans who are from the United States, while some Black immigrants or their children may also come to identify as African-American.
    The police are a constituted body of persons empowered by a state, with the aim to enforce the law, to ensure the safety, health and possessions of citizens, and to prevent crime and civil disorder. Their lawful powers include arrest and the use of force legitimized by the state via the monopoly on violence.
    Result: Hateful
    ###
    Meme Caption: '{meme_caption}'
    Image Caption: {image_caption}
    Context: {context}
    Result:"""
    parameters = {
        'max_new_tokens': 10,  
        'temperature': 1, 
        'end_sequence': "###"
    }

    options={'use_cache': True}
    result = mc.query(prompt,parameters, options)

    print(result)

    result = result[result.rfind('Result:'), result.rfind('###')]

    return f"Result= {result}"
