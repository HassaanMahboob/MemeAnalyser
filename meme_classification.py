import torch
import numpy as np
from fairseq import utils,tasks
from fairseq import checkpoint_utils
from utils.eval_utils import eval_step
from tasks.mm_tasks.caption import CaptionTask
from models.ofa import OFAModel
from PIL import Image
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import pytesseract
import tagme
nltk.download('punkt')
# Image transform
from torchvision import transforms

# Register caption task

def encode_text(text, bos_item, eos_item, length=None, append_bos=False, append_eos=False):
    s = task.tgt_dict.encode_line(
        line=task.bpe.encode(text),
        add_if_not_exist=False,
        append_eos=False
    ).long()
    if length is not None:
        s = s[:length]
    if append_bos:
        s = torch.cat([bos_item, s])
    if append_eos:
        s = torch.cat([s, eos_item])
    return s

# Construct input for caption task
def construct_sample(image: Image, pad_idx, bos_item, eos_item, patch_resize_transform):
    patch_image = patch_resize_transform(image).unsqueeze(0)
    patch_mask = torch.tensor([True])
    src_text = encode_text(" what does the image describe?", bos_item, eos_item,  append_bos=True, append_eos=True).unsqueeze(0)
    src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
    sample = {
        "id":np.array(['42']),
        "net_input": {
            "src_tokens": src_text,
            "src_lengths": src_length,
            "patch_images": patch_image,
            "patch_masks": patch_mask
        }
    }
    return sample
  
# Function to turn FP32 to FP16
def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

import matplotlib.pyplot as plt
import keras_ocr
import cv2
import math
import numpy as np
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)

def inpaint_text(img_path, pipeline):
    # read image
    img = keras_ocr.tools.read(img_path)
    # generate (word, box) tuples 
    prediction_groups = pipeline.recognize([img])
    mask = np.zeros(img.shape[:2], dtype="uint8")
    for box in prediction_groups[0]:
        x0, y0 = box[1][0]
        x1, y1 = box[1][1] 
        x2, y2 = box[1][2]
        x3, y3 = box[1][3] 
        
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        
        cv2.line(mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,    
        thickness)
        img = cv2.inpaint(img, mask, 7, cv2.INPAINT_NS)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_path,img_rgb)
    return(img)

def init_ocr_model():
    pipeline = keras_ocr.pipeline.Pipeline()
    return pipeline
def init_caption_model():
    tasks.register_task('caption',CaptionTask)
    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    # Load pretrained ckpt & config
    overrides={"bpe_dir":"utils/BPE", "eval_cider":False, "beam":5, "max_len_b":16, "no_repeat_ngram_size":3, "seed":7}
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths('checkpoints/caption.pt'),
            arg_overrides=overrides
        )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)


    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()
    return task, generator, models, use_cuda, use_fp16, pad_idx, bos_item, eos_item, patch_resize_transform

def get_caption(img_path, pipeline, task, generator, model, use_cuda, use_fp16, pad_idx, bos_item, eos_item, patch_resize_transform):
    inpaint_text(img_path, pipeline)
    image = Image.open(img_path)

    # Construct input sample & preprocess for GPU if cuda available
    sample = construct_sample(image, pad_idx, bos_item, eos_item, patch_resize_transform)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample
    with torch.no_grad():
        result, scores = eval_step(task, generator, model, sample)
    return result[0]['caption']

from SPARQLWrapper import SPARQLWrapper, JSON
from urllib.parse import urlparse,urlunparse
import os.path
import re
import requests



DBPEDIA_API_ENDPOINT = "http://lookup.dbpedia.org/api/search"




def init_engine():
    sparqldbpedia = SPARQLWrapper("https://dbpedia.org/sparql")
    sparqldbpedia.setReturnFormat(JSON)
    return sparqldbpedia

def get_dbpedia_description(engine,entity):
    engine.setQuery(f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX : <http://dbpedia.org/resource/>
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX dbpedia: <http://dbpedia.org/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    SELECT ?description WHERE
    {{
    <{entity}> dbo:abstract ?description.
    FILTER (lang(?description) = 'en')
    }}
    LIMIT 1
    """
    )
    result = engine.queryAndConvert()
    return result

def get_context_by_name(engine, entity):
    params_dbpedia = {
        'query':entity,
        'format':'JSON',
        'maxResults':1
        }

    #get the entities' dbpedia uri
    r = requests.get(url = DBPEDIA_API_ENDPOINT, params=params_dbpedia)
    data = r.json()
    try:
        urifromdbpedia = data["docs"][0]['resource'][0]
    except IndexError:
        #no dbpedia entry found, skipping dbpedia search
        pass
    #get the entities' description
    description_result = get_dbpedia_description(engine, urifromdbpedia)
    context = description_result["results"]["bindings"][0]["description"]["value"]
    return context

def get_context_by_id(engine, entity_id):
    engine.setQuery(f"""
    PREFIX owl: <http://www.w3.org/2002/07/owl#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX foaf: <http://xmlns.com/foaf/0.1/>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    PREFIX : <http://dbpedia.org/resource/>
    PREFIX dbpedia2: <http://dbpedia.org/property/>
    PREFIX dbpedia: <http://dbpedia.org/>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX dbo: <http://dbpedia.org/ontology/>
    SELECT ?description
    WHERE
    {{
    ?x dbo:wikiPageID {entity_id} .
    ?x dbo:abstract ?description.
    FILTER (lang(?description) = 'en')
    }}
    LIMIT 1
    """
    )
    result = engine.queryAndConvert()
    return result

def query(payload='',parameters=None, options={'use_cache': False}):
        API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-neo-2.7B"
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        body = {"inputs":payload,'parameters':parameters,'options':options}
        response = requests.request("POST", API_URL, headers=headers, data= json.dumps(body))
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError:
            return "Error:"+" ".join(response.json()['error'])
        else:
            return response.json()[0]['generated_text']
            
def OCR_img2txt(img_path):
    pytesseract.pytesseract.tesseract_cmd = r"C:\\tess\\tesseract.exe"
    return pytesseract.image_to_string(Image.open(img_path))

if __name__ == "__main__":
    img_path = "test.png"
    task, generator, models, use_cuda, use_fp16, pad_idx, bos_item, eos_item, patch_resize_transform = init_caption_model()
    pipeline = init_ocr_model()
    meme_caption = OCR_img2txt(img_path)
    image_caption = get_caption(img_path, pipeline, task, generator, models, use_cuda, use_fp16, pad_idx, bos_item, eos_item, patch_resize_transform)
    context = ""
    tagme.GCUBE_TOKEN = "dbc68d3b-0127-4709-9608-761d8e6c585f-843339462"
    enginedbpedia = init_engine()
    annotations = tagme.annotate(meme_caption)
    unique_entities = set()
    for entity in annotations.get_annotations(0.07):
        unique_entities.add(entity.entity_id)
        print(entity.entity_title)
    unique_entities = list(unique_entities)
    for entity_id in unique_entities:
        desc = get_context_by_id(enginedbpedia, str(entity_id))["results"]["bindings"][0]["description"]["value"]
        desc = sent_tokenize(desc)
        desc = ' '.join(desc[0:2])
        re.sub("[\(\[].*?[\)\]]", "", desc)
        context+= desc
        if entity_id != unique_entities[-1]:
            context+="\n"
    
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

    API_TOKEN = "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"

    import json
    

    parameters = {
        'max_new_tokens': 20,  
        'temperature': 1, 
        'end_sequence': "###"
    }

    options={'use_cache': True}
    result = query(prompt,parameters, options)

    print(result)


