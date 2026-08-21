# Predicting User Behavior Using Machine Learning and LSTM
Ky punim paraqet implementimin dhe eksperimentet për parashikimin e ndërveprimit pasues të përdoruesve bazuar në sekuencat e ndërveprimeve të mëparshme në një dataset e-commerce.
Në këtë punim është përdorur dataset-i **E-commerce Behavior Data from Multi Category Store**, ndërsa për eksperimentet është përdorur skedari **2019-Oct.csv** nga i cili janë përzgjedhur 6,000,000 regjistime.
Ndërveprimet e analizuara janë:
- view
- cart
- purchase
Për parashikimin e ndërveprimit pasues janë krijuar sekuenca me gjatësi pesë, ku pesë ndërveprimet e mëparshme përdoren për të parashikuar ndërveprimin e radhës.
Në punim janë implementuar dhe krahasuar tre modele:
- Logistic Regression
- Decision Tree
- Long Short-Term Memory (LSTM)
Gjithashtu është analizuar ndikimin i pabalancimit të klasave dhe është përdorur Random UnderSampling për balancimin e të dhënave të trajnimit.

 ## Dataset-i
 Në këtë punim është përdorur dataset-i **E-commerce Behavior Data from Multi Category Store** i publikuar në Kaggle nga mkechinov.
 Për eksperimentet është përdorur skedari `2019-Oct.csv`. Për shkak të madhësisë së dataset-it origjinal, në këtë punim janë përdorur 6,000,000 regjistime, të lexuara në mënyrë të ndarë përmes chunks.

 Dataset-i përmban 9 kolona kryesore:
 - event_time - koha e ndërveprimit
 - event_type - lloji i ndërveprimit
 - product_id - identifikuesi i produktit
 - category_id - identifikuesi i kategorisë
 - category_code - kodi i kategorisë
 - brand - marka e produktit
 - price - çmimi i produktit
 - user_id - identifikuesi i përdoruesit
 - user_session - identifikuesi i sesionit të përdoruesit

Në analizë janë përdorur tre lloje ndërveprimesh: view, cart dhe purchase.

## Data Preprocessing
Para ndërtimit të sekuencave dhe trajnimit të modeleve, të dhënat iu nënshtruan disa hapave të përpunimit.

### Removing Duplicate Records
Nga 6,000,000 regjistrime fillestare u identifikuan dhe u larguan 3,779 të duplikuara. Pas këtij procesi, dataset-i përmbante 5,996,221 regjistrime.

### Handling Missing Values
Vlerat që mungonin në kolonat brand dhe category_code u zëvendësuan me vlerën "unknown". Kjo mundësoi që të mos mbeteshin vlera të zbrazëta në këto kolona gjatë analizës së mëtejshme.

### Converting Event Time
Kolona evet_time u konvertua në formatin datatime duke përdorur UTC, në mënyrë që informacioni kohor të mund të përdorej në mënyrë të saktë gjatë organizimit të ndërveprimeve.

### Chronological Sorting
Të dhënat u rënditën sipas user_session dhe event_time, duke siguruar që ndërveprimet brenda secilit sesion të ishin në rend kronologjik.

### Encoding Event Types
Llojet e ndërveprimeve u shndërruan në vlera numerike:
- view -> 0
- cart -> 1
- purchase -> 2
Pas kodimit nuk u identifikuan vlera që mungonin në kolonën event_code.

### Class Distribution
Pas pastrimit të të dhënave, shpërndarja e ndërveprimeve ishte:
- view : 5,767,762 (96.19%)
- cart: 123,581 (2.06%)
- purchase : 104,888 (1.75%)
Kjo shpërndarje tregon një pabalancim të konsiderueshëm ndërmjet klasave, ku view përbën shumicën dërrmuese të ndërveorimeve.

## Sequential Data Preparation
Për të mundësuar parashikimin e ndërveprimit pasues të përdoruesit, ndërveprimet u organizuan në formë sekuencash bazuar në user_session.
Pas analizës së sesioneve, u identifikuan 1,289,469 sesione unike. Gjatësia mesatare e një sesioni ishte 4.65 ndërveprime, ndërsa 309,428 sesione përmbanin të paktën 6 ndërveprime dhe mund të përdoreshin për krijimin e sekuencave.
Në këtë punim u përdor një gjatësi sekuence prej 5 ndërveprimesh. Kjo do të thotë se pesë ndërveprimet e mëparshme të një përdoruesi u përdorën si hyrje për të parashikuar ndërveprimin pasues.
Në total u krijuan 2,360,318 sekuenca. Të dhënat e hyrjes `X` kanë formën: `(2,360,318, 5)`, ndërsa variabla objektive `y` ka formën: `(2,360,318, )`
Çdo sekuencë përfaqëson pesë ndërveprime të njëpasnjëshme, ndërsa vlera e objektivit përfaqëson ndërveprimin që vjen menjëherë pas tyre.
Kodimi i përdorur për ndërveprimet ishte:
- view -> 0
- cart -> 1
- purchase -> 2
Shpërndarja e targeteve në sekuencat e krijuara ishte:
- view : 2,278,353 (96.53%)
- cart : 43,274 (1.83%)
- purchase : 38,691 (1.64%)
Pas krijimit të sekuencave, ato u rënditën në mënyrë kronologjike sipas kohës së ndërveprimit target, për të ruajtur rendin kohor të të dhënave.

## Train/Test Split
Pas krijimit dhe renditjes kronologjike të sekuencave, të dhënat u ndanë në dy pjesë: të dhënat për trajnim dhe të dhënat për testim.
Për të shmangur përdorimin e informacionit nga e ardhmja gjatë trajnimit, u përdor një ndarje kronologjike e të dhënave. Reth 80% e sekuencave u përdorën për trajnim, ndërsa 20% u përdorën për testim.
Nga gjithsej 2,360,318 sekuenca:
- Training: 1,888,253 sekuenca (80%)
- Testing: 472,065 sekuenca (20%)
Ndarja u realizua duke përdorur kohën e ndërveprimit target. Koha e ndarjes ishte:
`2019-10-04 17:55:35 UTC`
Kështu, të dhënat e trajnimit përfshijnë ndërveprimet target nga:
`2019-10-01 00:02:41 UTC` deri në `2019-10-04 17:55:34 UTC`
Ndërsa të dhënat e testimit përfshijnë ndërveprimet target nga:
`2019-10-04 17:55:35 UTC` deri në `2019-10-05 16:25:48 UTC`
U verifikua që:
- rendi kronologjik i të dhënave të trajnimit është ruajtur;
- rendi kronologjik i të dhënave të testimit është ruajtur;
- nuk ka mbivendosje kohore ndërmjet train dhe test;
- numri total i sekuencave është ruajtur pas ndarjes.

## Experiments Before Class Balancing

Para aplikimit të metodës së balancimit të klasave, tre modele u trajnuan dhe u testuan duke përdorur të dhënat në shpërndarjen e tyre origjinale.

### LSTM
Modeli Long Short-Term Memory (LSTM) u ndërtua për të përpunuar sekuencat prej pesë ndërveprimesh dhe për të parashikuar ndërveprimin pasues.
Arkitektura e modelit përfshinte:
- Embedding layer me dimension 8;
- LSTM layer me 32 njësi;
- Dropout me normë 0.2;
- Dense layer me 16 njësi dhe aktivizim ReLU;
- Output layer me 3 klasa dhe aktivizim Softmax.
Modeli u trajnua për 5 epochs, me batch size 256 dhe pa përzierje të të dhënave gjatë trajnimit.
Në testim, LSTM arriti:
- Accuracy: 95.91%
- Precision: 94.78%
- Recall: 95.91%
- F1-Score: 94.45%
- Macro F1-Score: 45.71%
Megjithatë, performanca ndryshonte dukshëm ndërmjet klasave. Recall ishte 1.00 për `view`, 0.08 për `cart` dhe 0.15 për `purchase`.

### Logistic Regression
Logistic Regression u përdor si një model krahasues tradicional për klasifikimin e sekuencave të ndërveprimeve.
Modeli arriti:
- Accuracy: 95.41%
- Precision: 92.96%
- Recall: 95.41%
- F1-Score: 93.61%
- Macro F1-Score: 36.14%
Performanca sipas klasave tregoi se modeli identifikonte shumë mirë klasën `view`, ndërsa kishte vështirësi në identifikimin e `cart` dhe `purchase`.
Recall ishte 1.00 për `view`, 0.05 për `cart` dhe 0.01 për `purchase`.

### Decision Tree
Decision Tree u përdor si model tjetër krahasues për klasifikimin e ndërveprimeve pasuese.
Modeli arriti:
- Accuracy: 95.90%
- Precision: 94.58%
- Recall: 95.90%
- F1-Score: 94.59%
- Macro F1-Score: 47.28%
Në krahasim me Logistic Regression dhe LSTM, Decision Tree tregoi performancë më të mirë në identifikimin e klasave `cart` dhe `purchase`, megjithëse edhe te ky model dominonte klasa `view`.
### Model Comparison Before Balancing

Rezultatet para balancimit treguan se të tre modelet kishin Accuracy të lartë, me rezultate shumë të afërta:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Logistic Regression | 95.41% | 0.3614 |
| Decision Tree | 95.90% | 0.4728 |
| LSTM | 95.91% | 0.4571 |

Megjithëse Accuracy ishte mbi 95% për të tre modelet, Macro F1 ishte dukshëm më i ulët. Kjo tregon se Accuracy ishte ndikuar nga dominimi i klasës `view` dhe nuk e pasqyronte plotësisht performancën e modeleve në klasat `cart` dhe `purchase`.

### Confusion Matrix Before Balancing
Confusion Matrix u përdor për të analizuar më në detaje klasifikimet e sakta dhe gabimet e secilit model.
Analiza tregoi se shumica e rasteve reale të klasave `cart` dhe `purchase` klasifikoheshin gabimisht si `view`. Kjo e konfirmoi ndikimin e pabalancimit të klasave në performancën e modeleve.

## Class Balancing – Random UnderSampling
Për shkak të pabalancimit të konsiderueshëm të klasave në të dhënat e trajnimit, u aplikua metoda Random UnderSampling. Qëllimi i këtij procesi ishte reduktimi i dominimit të klasës shumicë `view` dhe krijimi i një shpërndarjeje më të ekuilibruar të klasave gjatë trajnimit.
Para balancimit, të dhënat e trajnimit përmbanin:

| Event | Count | Percentage |
|---|---:|---:|
| view | 1,827,167 | 96.76% |
| cart | 31,169 | 1.65% |
| purchase | 29,917 | 1.58% |

Me anë të Random UnderSampling, numri i mostrave të klasës shumicë `view` u reduktua për t'u përshtatur me klasën më të vogël `purchase`.
Pas balancimit, secila klasë përmbante 29,917 mostra:

| Event | Count | Percentage |
|---|---:|---:|
| view | 29,917 | 33.33% |
| cart | 29,917 | 33.33% |
| purchase | 29,917 | 33.33% |

Si rezultat, madhësia e të dhënave të trajnimit u reduktua nga 1,888,253 në 89,751 sekuenca.
Të dhënat e testimit nuk u balancuan dhe mbetën të pandryshuara, në mënyrë që performanca e modeleve të vlerësohej mbi shpërndarjen reale të të dhënave të testimit.
Balancimi u aplikua vetëm në të dhënat e trajnimit.

## Experiments After Class Balancing
Pas aplikimit të Random UnderSampling, tre modelet u trajnuan përsëri duke përdorur të dhënat e balancuara të trajnimit. Të dhënat e testimit mbetën të pandryshuara.

### LSTM After Balancing
LSTM u trajnua me të njëjtën arkitekturë si në skenarin para balancimit. Të dhënat e balancuara përmbanin 89,751 sekuenca trajnimi, me shpërndarje të barabartë ndërmjet tri klasave.
Modeli u trajnua për 5 epochs, me batch size 256 dhe me përzierje (`shuffle=True`) gjatë trajnimit.
Në testim, LSTM arriti:
- Accuracy: 91.58%
- Precision: 95.14%
- Recall: 91.58%
- F1-Score: 93.09%
- Macro F1-Score: 0.5518
Recall sipas klasave ishte:
- `view`: 0.93
- `cart`: 0.52
- `purchase`: 0.55
Krahasuar me skenarin para balancimit, Recall për klasat `cart` dhe `purchase` u rrit ndjeshëm, ndërsa Accuracy e përgjithshme u ul.

### Logistic Regression After Balancing
Logistic Regression u trajnua përsëri duke përdorur të dhënat e balancuara.
Rezultatet në të dhënat e testimit ishin:
- Accuracy: 90.43%
- Precision: 94.94%
- Recall: 90.43%
- F1-Score: 92.35%
- Macro F1-Score: 0.5122
Recall sipas klasave ishte:
- `view`: 0.92
- `cart`: 0.47
- `purchase`: 0.51
Balancimi përmirësoi identifikimin e klasave `cart` dhe `purchase`, megjithëse pati një ulje të Accuracy krahasuar me skenarin para balancimit.

### Decision Tree After Balancing
Decision Tree u trajnua duke përdorur të njëjtat të dhëna të balancuara të trajnimit.
Modeli arriti:
- Accuracy: 91.76%
- Precision: 95.16%
- Recall: 91.76%
- F1-Score: 93.19%
- Macro F1-Score: 0.5540
Recall sipas klasave ishte:
- `view`: 0.94
- `cart`: 0.51
- `purchase`: 0.56
Decision Tree arriti rezultatet më të larta të përgjithshme ndërmjet tre modeleve në skenarin pas balancimit.

### Model Comparison After Balancing

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| Logistic Regression | 90.43% | 92.35% | 0.5122 |
| Decision Tree | 91.76% | 93.19% | 0.5540 |
| LSTM | 91.58% | 93.09% | 0.5518 |

Rezultatet tregojnë se Decision Tree kishte performancën më të lartë të përgjithshme pas balancimit, ndërsa LSTM ishte shumë afër tij.
Në krahasim me skenarin para balancimit, të tre modelet patën ulje të Accuracy, por përmirësuan performancën në klasat më pak të përfaqësuara.

### Minority Class Recall
Balancimi ndikoi dukshëm në Recall të klasave `cart` dhe `purchase`.

| Model | Cart Recall Before | Cart Recall After | Purchase Recall Before | Purchase Recall After |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.05 | 0.47 | 0.01 | 0.51 |
| Decision Tree | 0.11 | 0.51 | 0.16 | 0.56 |
| LSTM | 0.08 | 0.52 | 0.15 | 0.55 |

Këto rezultate tregojnë se Random UnderSampling ka përmirësuar ndjeshëm aftësinë e modeleve për të identifikuar ndërveprimet `cart` dhe `purchase`.

### Confusion Matrix After Balancing
Confusion Matrix pas balancimit tregon një shpërndarje më të mirë të parashikimeve ndërmjet klasave krahasuar me skenarin para balancimit.
Edhe pse një numër i konsiderueshëm i rasteve `cart` dhe `purchase` vazhdojnë të klasifikohen si `view`, numri i klasifikimeve të sakta për të dyja klasat u rrit dukshëm.
Në veçanti, për klasën `purchase`, Decision Tree klasifikoi saktë 4,930 raste, ndërsa LSTM klasifikoi saktë 4,822 raste. Për klasën `cart`, LSTM klasifikoi saktë 6,238 raste, ndërsa Decision Tree 6,120 raste.

### Final Results
Krahasimi përfundimtar u realizua duke analizuar Accuracy dhe Macro F1-Score para dhe pas balancimit të klasave.

| Model | Accuracy Before | Accuracy After | Macro F1 Before | Macro F1 After |
|---|---:|---:|---:|---:|
| Logistic Regression | 95.41% | 90.43% | 0.3614 | 0.5122 |
| Decision Tree | 95.90% | 91.76% | 0.4728 | 0.5540 |
| LSTM | 95.91% | 91.58% | 0.4571 | 0.5518 |

Rezultatet tregojnë se Random UnderSampling ka shkaktuar një ulje të Accuracy për të tre modelet, por njëkohësisht ka përmirësuar Macro F1-Score dhe Recall për klasat cart dhe purchase.
Pas balancimit, Decision Tree arriti Accuracy më të lartë, me 91.76%, dhe Macro F1-Score prej 0.5540. LSTM rezultoi shumë afër Decision Tree, me Accuracy 91.58% dhe Macro F1-Score prej 0.5518.
Në skenarin para balancimit, Accuracy e lartë e modeleve shoqërohej me performancë dukshëm më të ulët në klasat minoritare. Prandaj, krahasimi i rezultateve pas balancimit tregon rëndësinë e përdorimit të metrikave që marrin parasysh performancën në të gjitha klasat.

## Evaluation Metrics
Për vlerësimin e modeleve janë përdorur metrikat Accuracy, Precision, Recall, F1-Score dhe Macro F1-Score.
Accuracy – tregon përqindjen e parashikimeve të sakta në të gjitha rastet.
Precision – tregon sa nga parashikimet e një klase kanë qenë të sakta.
Recall – tregon sa nga rastet reale të një klase janë identifikuar saktë.
F1-Score – kombinon Precision dhe Recall në një metrikë të vetme.
Macro F1-Score – llogarit F1-Score veçmas për secilën klasë dhe më pas merr mesataren e tyre, duke i dhënë të njëjtën peshë secilës klasë.
Përveç këtyre metrikave, është përdorur edhe Confusion Matrix për të analizuar klasifikimet e sakta dhe llojet e gabimeve të bëra nga modelet.

## Technologies
Projekti është realizuar duke përdorur:
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- TensorFlow / Keras
- Kaggle Notebook
- GitHub

## How to Run
Notebook-u i projektit gjendet në:
notebooks/user_behavior_prediction_lstm.ipynb
Kodi është zhvilluar në mjedisin Kaggle dhe përdor dataset-in 2019-Oct.csv nga dataset-i E-commerce Behavior Data from Multi-Category Store.
Për ekzekutimin e notebook-ut duhet të jetë i disponueshëm dataset-i dhe të përshtatet rruga e dataset-it sipas mjedisit ku ekzekutohet kodi.

Notebook-u përmban të gjithë procesin e projektit, duke përfshirë:
- ngarkimin e dataset-it;
- analizën eksploruese të të dhënave;
- pastrimin dhe përpunimin e të dhënave;
- krijimin dhe validimin e sekuencave;
- ndarjen kronologjike në train dhe test;
- trajnimin e modeleve para balancimit;
- aplikimin e Random UnderSampling;
- trajnimin e modeleve pas balancimit;
- vlerësimin e performancës;
- krahasimin përfundimtar të modeleve.

