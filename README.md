# Age Prediction from Faces (CNN)

A small deep learning project: a convolutional neural network (CNN) that looks at a face photo and predicts the person's **age**, plus a Streamlit app where you can take a photo with your camera or upload one.

**Live app:** [APP_LINK](https://age-prediction-conv-nn.streamlit.app/)

## Dataset
[UTKFace](https://www.kaggle.com/datasets/jangedoo/utkface-new): 23,708 cropped face photos with the age in the file name (`age_gender_race_date.jpg`).
- ages from 1 to 116
- images are resized to 64×64 RGB
- split: 15,172 train / 3,794 validation / 4,742 test

The data is **not included in this repo** (too big). Download it from Kaggle and put it like this:

```
data/
  UTKFace/   (all the .jpg files)
```

## Project structure
```
requirements.txt          packages for the app
code/
  cnn_miniproject.ipynb   loading, training, evaluation, error analysis
  age_app.py              Streamlit app (camera + upload)
  age_model.keras         saved model used by the app
data/                     (not in the repo, see above)
```

## Approach
1. **Load:** read every image, resize to 64×64, take the age from the file name.
2. **Split:** train / validation / test (64% / 16% / 20%).
3. **Baseline:** always predicting the average age.
4. **First CNN:** 3 conv blocks (32-64-128) with `MaxPooling(3,3)`, Dense 1024, 1 output. MAE loss, Adam, early stopping on `val_mae` (patience 3).
5. **Better CNN (`imodel`, used in the app):** same idea but `MaxPooling(2)` with `padding='same'`, so the feature map ends at 8×8×128 instead of 1×1×128.
6. **Error analysis:** actual vs predicted age, MAE per age group.

## Results (test set, MAE in years)
| Model | MAE |
|---|---|
| Baseline (average age) | 15.42 |
| First CNN (pooling 3) | 7.31 |
| `imodel` (pooling 2), used in the app | **6.84** |

MAE per age group (`imodel`):

| Age group | MAE | Test images |
|---|---|---|
| 0-10 | 4.3 | 602 |
| 10-20 | 7.4 | 284 |
| 20-30 | 4.8 | 1489 |
| 30-40 | 5.6 | 883 |
| 40-50 | 7.9 | 438 |
| 50-60 | 10.1 | 477 |
| 60-80 | 12.1 | 428 |
| 80-120 | 15.3 | 141 |

The error grows with age. There are few old faces in the data, so the model probably pulls its predictions toward the middle ages.

## Run it yourself
```bash
pip install -r requirements.txt
streamlit run code/age_app.py
```
Run it from the main project folder (the one that contains `code/` and `requirements.txt`).
To run the notebook also install: `pip install pandas matplotlib scikit-learn jupyter`

## Limitations and ethics
- **Average error is about 7 years**, and much more for older people, so it must not be used for real decisions (for example age verification).
- **Cropped faces only:** the model was trained on close-up cropped faces. For the camera, fill the frame with your face; a full photo with a background gives bad results.
- **Image quality:** 64×64 only; bad lighting, angles, glasses or hats hurt it.
- **Dataset bias:** UTKFace is not balanced across ages, and the balance of gender and ethnicity is not guaranteed, so it may work worse for some groups.
- **Privacy and misuse:** faces are personal data and need consent. Photos in the app are only used for the prediction. Not for surveillance, hiring or profiling people.
