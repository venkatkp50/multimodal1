
import io
import json
import time
import math
import requests
import rouge
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from nltk.corpus import stopwords
import seaborn as sns
import matplotlib.pyplot as plt
import re, os, string, random, requests
#from summarizer import Summarizer,TransformerSummarizer
#from bert_score import score
import plotly.graph_objects as go
import plotly.express as px
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate import meteor


from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
from IPython.display import display, Image as IPImage

from multimodalReader import getImage


import warnings
warnings.filterwarnings('ignore')


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
new_stopwords = ["What"]
STOPWORDS = set(stopwords.words('english'))

BERT_MAX_TOKEN = 512
GPT2_MAX_TOKEN = 1024
doc_dir = 'text_file'
content_image ='images\content'
content_image_width = 250

import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")
imagename = Image.open('images/caronavirus banner.jpg')



import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

rouge = rouge.Rouge()
imagename2 = Image.open('images/Sidebar2.jpg')
st.sidebar.image(imagename2)
st.sidebar.title('Settings')
modelSelected = st.sidebar.selectbox('Choose Reader Model',options=('deepset/roberta-base-squad2-covid','deepset/roberta-base-squad2','deepset/covid_bert_base'))

# mystyle = '''
#     <style>
#         p {
#             text-align: justify;
#         }
#     </style>
#     '''
def cleanText(text):  
  text = re.sub(r"[^a-zA-Z]"," ",text)  
  txt = [wrd.lower() for wrd in word_tokenize(text) if wrd.lower() not in STOPWORDS]
  txt  = ' '.join(w for w in txt)
  return txt

def rerun():
    file2 = open("sessioncount.txt","w+")
    file2.write('0')
    file2.close()
    st.session_state.input_text = ''
    #st.experimental_rerun()

def getTextfromList(text):
    header =[]
    para = []
    for line in sent_tokenize(text):
        if not line is np.nan:
            if len(line) > 1:
                if len(line) < 100:
                    header.append(line)
                else:
                    para.append(line)
                    
    text =  ''.join(line for line in para)
    return text


tab1,  tab3 = st.tabs(["Single Document Summarization","Multi Document Summarization"])
    with tab1:
        st.write('inside tab1 .................')
    with tab3:
            st.subheader('Content similarity between top files')
#             polarity = []
#             subjectivity = []
#             senti = []
#             similarity_score = []
#             cosine_similarity = []
#             jaccard_similarity = []
#             text1 = data[data['paper_id'] == ids[0].replace('.txt','')]['text'].values[0]                
#             text1Str = getTextfromList(text1)
            
#             uniqueID = list(set(ids))
           
#             for id in uniqueID: 
#                 text2 = data[data['paper_id'] == id.replace('.txt','')]['text'].values[0]
#                 text2Str = getTextfromList(text2)                 
#                 pol, sub = getSentiment(text2)

#                 cosine_similarity.append(getcosineSimilarity(text1Str,text2Str))
#                 jaccard_similarity.append(getjaccardSimilarity(text1Str,text2Str))
#                 similarity_score.append( getSimilarityScore(text1Str,text2Str))
#                 polarity.append(pol)
#                 subjectivity.append(sub)

#             df3 = pd.DataFrame({'File_name':uniqueID,'Polarity':polarity,'Subjectivity':subjectivity,'BERT Similarity':similarity_score,'Cosine similarity ':cosine_similarity,'Jaccard Similarity':jaccard_similarity})
#             st.table(df3)
#             st.markdown('------')
#             st.subheader('Key sentences from top files')
#             multidocsummary = []
#             for id in uniqueID:
#                 text = data[data['paper_id'] == id.replace('.txt','')]['text'].values[0]
#                 for highlight in analyze(text):
#                     if not highlight in multidocsummary:
#                         multidocsummary.append(highlight)

#             count = 1
#             imageList = []
#             for txt in multidocsummary:
#                 st.write(str(count),'. ',txt.capitalize())
# #                 imageKeyWords = ' '.join(wrd for wrd in get_keywords(cleanText(txt)))
                
#                 images_array, imgscore = getImage('cold')                
#                 imageName  = images_array[0]
#                 #st.write(imgscore,imageList)
#                 if not imageName in imageList:
#                     if float(imgscore[0]) > 0.5:
#                         imagenamex = Image.open(imageName,"r")
#                         st.image(imagenamex,width=content_image_width)
#                         imagenamex.close()
#                         imageList.append(imageName)
#                         st.write(imageName)
# #                         st.write(','.join(wrd.upper() for wrd in word_tokenize(imageKeyWords)))
#                         #st.write(getDispacy(imageKeyWords.upper()))

#                 count += 1



#             # images_array, imgscore = getImage(user_message)
#             # # st.write(images_array,imgscore)
#             # imagenamex = Image.open(images_array[0],"r")
#             # st.image(imagenamex,width=content_image_width)
#             # imagenamex.close()
#             #for ima, score in zip(images_array, imgscore):
#                 # im = Image.open(ima)
#                 # img_with_border = ImageOps.expand(im, border=20, fill="white")
#                 # img = ImageDraw.Draw(img_with_border)
#                 # img.text((20, 0), f"Score: {score},    Path: {ima}", fill=(0, 0, 0))
#                 # bio = BytesIO()
#                 # img_with_border.save(bio, format="png")   
#                 # plt.show(display(IPImage(bio.getvalue(), format="png",width = 200)))

# #             getDispacy('test case for')
# #             st.markdown('----')

# #         new_query = st.button('New Query',on_click=rerun)




# # doc_dir = 'text_file'
# # file1 = open("sessioncount.txt","r")
# # runtime = file1.read()
# # print('2................',runtime)
# # file1.close()
# # if runtime == '0':
# #     st.image(imagename)
# #     textinput = st.text_input("Your Query", key="input_text",value='',on_change=runSumm)    


    




