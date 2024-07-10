import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel

# Data load
imagenette = load_dataset(
    'frgfm/imagenette',
    '320px',
    split='validation',
    revision="4d512db"
)

model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Text 라벨링 
labels = imagenette.info.features['label'].names
clip_labels = [f"a photo of a {label}" for label in labels]

# Text encoding
label_tokens = processor(
    text=clip_labels,
    padding=True,
    images=None,
    return_tensors='pt'
).to(device)

# encode tokens to sentence embeddings
label_emb = model.get_text_features(**label_tokens)
# detach from pytorch gradient computation
label_emb = label_emb.detach().cpu().numpy()
# normalization
label_emb = label_emb / np.linalg.norm(label_emb, axis=0)
label_emb.min(), label_emb.max()



#Image
preds = []  #높은 값 인덱스
scorelist = []  #모든 스코어 값 저장
scoresort = []  #스코어 정렬
threepreds = [] #유사도 3,2,1위 인덱스
labellist = []  #유사도 1,2,3위 text
batch_size = 32

for i in tqdm(range(0, len(imagenette), batch_size)):
    i_end = min(i + batch_size, len(imagenette))  
    images = processor(
        text=None,
        images=imagenette[i:i_end]['image'],
        return_tensors='pt'
    )['pixel_values'].to(device)
    img_emb = model.get_image_features(images)
    img_emb = img_emb.detach().cpu().numpy()
    scores = np.dot(img_emb, label_emb.T)
    
    preds.extend(np.argmax(scores, axis=1))  #스코어 중 가장 높은 값의 인덱스 뽑는다.
    scorelist.extend(scores)
    scoresort.extend(np.argsort(scores))
    
threepred = [st[-3:] for st in scoresort]
labellist = [(labels[k], labels[j], labels[i]) for i, j, k in threepred]    


#성능
true_preds = []
for i, label in enumerate(imagenette['label']):
    if label == preds[i]:
        true_preds.append(1)
    else:
        true_preds.append(0)
print(f'Performance : {(sum(true_preds) / len(true_preds))}') 




#엑셀로 저장
lable_text = [labels[i] for i in imagenette['label']]  #정답 text
preds_text = [labels[i] for i in preds]  # 예측 text

raw_data = {'정답' : lable_text ,
            '예측' : preds_text,
            '유사도 순위' : labellist            
            } # 리스트 자료형으로 생성

raw_data = pd.DataFrame(raw_data) # 데이터 프레임으로 전환
raw_data.to_excel(excel_writer='sample.xlsx') # 엑셀로 저장