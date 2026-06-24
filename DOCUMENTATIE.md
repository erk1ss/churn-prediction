# Concluzii EDA — Telco Customer Churn

## Data Cleaning

**Problema `TotalCharges`:** coloana era citită ca tip `string` în loc de numeric, din cauza a 11 valori care conțineau doar spațiu gol (`" "`). Investigând cauza, am observat că aceste 11 rânduri corespund clienților cu `tenure = 0` (clienți noi, neîncă facturați) — deci nu erau erori de date, ci o consecință logică. Am convertit coloana cu `pd.to_numeric(..., errors='coerce')` și am completat valorile lipsă rezultate cu `0` (`fillna(0)`), justificat de faptul că un client cu `tenure = 0` nu a acumulat încă nicio factură.

## Distribuția target-ului (Churn)

Dataset-ul e **dezechilibrat**: 73.5% "No" vs 26.5% "Yes". Această observație e importantă pentru etapa de modelare — un model care prezice mereu "No" ar avea 73.5% acuratețe fără să identifice niciun client la risc real, deci accuracy nu va fi metrica principală de evaluare; vom folosi precision, recall, F1-score și confusion matrix.

## Tenure (vechimea clientului) vs Churn

Clienții cu vechime mică (0-5 luni) au o rată de abandon mult mai mare comparativ cu clienții cu vechime mare. Rata de abandon scade constant odată cu creșterea `tenure`, iar clienții foarte vechi (~70+ luni) au o rată de abandon aproape nulă. **Concluzie:** primele luni sunt critice pentru retenție — companiile ar trebui să investească în experiența clienților noi.

## Tipul de contract vs Churn

- **Month-to-month:** ~43% rată de abandon (cea mai mare)
- **One year:** ~11% rată de abandon
- **Two year:** ~3% rată de abandon (aproape nulă)

**Concluzie:** lipsa unui angajament contractual pe termen lung e unul dintre cei mai puternici predictori ai abandonului. Combinat cu observația despre `tenure`, profilul de risc maxim este: client nou + contract month-to-month.

## MonthlyCharges vs Churn

Clienții care au renunțat la serviciu au, în medie, facturi lunare mai mari (mediană ~$80) comparativ cu cei care au rămas (mediană ~$65). Distribuția arată că riscul de abandon crește odată cu costul abonamentului — posibil din cauza sensibilității la preț sau a percepției unui raport cost-beneficiu nesatisfăcător. Această observație completează profilul de risc identificat anterior (client nou, contract month-to-month) cu un al treilea factor: cost lunar ridicat.

## PaymentMethod vs Churn

Clienții care plătesc prin "Electronic check" au o rată de abandon de ~45%, semnificativ mai mare comparativ cu celelalte metode de plată (Mailed check, Bank transfer automat, Credit card automat), care au rate de ~17-19%. Plățile automate (debit direct) par asociate cu o retenție mai bună, posibil din cauza lipsei de interacțiune activă lunară care ar putea declanșa decizia de anulare.

## Analiză sistematică completă (toate variabilele)

Pentru a evita o selecție arbitrară, am verificat automat rata de abandon pentru toate variabilele categorice și corelațiile pentru toate variabilele numerice.

**Variabile fără impact semnificativ:**
- `gender` — practic identic (26.9% vs 26.2% abandon) — genul nu influențează decizia de abandon
- `PhoneService` — diferență mică (24.9% vs 26.7%)

**Variabile noi cu impact notabil:**
- `Partner` — clienții fără partener au 33% abandon, vs 19.7% pentru cei cu partener
- `Dependents` — fără persoane în întreținere: 31.3% abandon, vs 15.5% cu persoane în întreținere
- Pattern general: stabilitatea în viața personală (relație, familie) corelează cu retenție mai bună

**Corelații numerice:**
- `tenure` și `TotalCharges` au corelație foarte mare (0.83) — multicoliniaritate, de luat în considerare la modelare
- `tenure` are cea mai puternică corelație cu `Churn` (-0.35), urmat de `MonthlyCharges` (0.19) și `SeniorCitizen` (0.15)

## Servicii de internet și adiționale vs Churn

- **InternetService:** Fiber optic are cea mai mare rată de abandon (41.9%), urmat de DSL (19%) și clienții fără internet (7.4%). Serviciul "premium" (fibră) e paradoxal asociat cu abandon mai mare, posibil din cauza prețului ridicat sau a problemelor de satisfacție specifice acestui tip de conexiune.
- **OnlineSecurity, OnlineBackup, DeviceProtection:** Pattern consistent — clienții fără aceste servicii adiționale au rate de abandon de 2-3x mai mari decât cei care le au (~40% vs ~15-22%). Clienții mai "investiți" în servicii suplimentare par mai loiali.

**Profil de risc actualizat (complet):** client nou, fără partener/dependenți, contract month-to-month, internet prin fibră optic, fără servicii de securitate/backup, factură lunară mare, plată prin electronic check.

## Preprocesare pentru ML

- Eliminat `customerID` (fără valoare predictivă) și coloana temporară `Churn_numeric`
- One-Hot Encoding pentru toate variabilele categorice (`pd.get_dummies`, `drop_first=True`) — dataset-ul a crescut de la 21 la 31 coloane
- Train/Test split: 80/20, cu `stratify=y` pentru a păstra proporția claselor (dezechilibrul 73.5%/26.5%) identică în ambele seturi
- Scalare cu `StandardScaler` pentru `tenure`, `MonthlyCharges`, `TotalCharges` — fit doar pe train, transform pe test (evităm data leakage)

## Primul model: Logistic Regression

Am antrenat un model de Logistic Regression ca baseline (model simplu, interpretabil, punct de referință pentru modele viitoare mai complexe).

### Interpretarea coeficienților

Coeficienții confirmă marea majoritate a observațiilor din EDA:
- **`InternetService_Fiber optic`: +1.18** (cel mai mare coeficient pozitiv) — confirmă riscul ridicat asociat fibrei optice
- **`tenure`: -1.26** — confirmă că vechimea reduce semnificativ riscul de abandon
- **`Contract_Two year`: -1.31** și **`Contract_One year`: -0.68** — confirmă că contractele lungi reduc riscul
- **`PaymentMethod_Electronic check`: +0.39** — confirmă riscul ridicat asociat acestei metode de plată

### Descoperire: Multicoliniaritate (TotalCharges, MonthlyCharges, tenure)

`MonthlyCharges` a apărut cu coeficient **negativ** (-0.47), contrazicând aparent corelația pozitivă observată în EDA (0.19). Cauza: **multicoliniaritate** — `tenure`, `MonthlyCharges` și `TotalCharges` sunt puternic corelate între ele (`tenure`-`TotalCharges`: 0.83), deoarece `TotalCharges` ≈ `tenure × MonthlyCharges`. Când variabile explicative sunt puternic corelate, modelul nu poate izola clar efectul individual al fiecăreia, ceea ce poate produce coeficienți individuali înșelători, chiar dacă predicțiile modelului per total rămân valide.

**Decizie:** eliminăm `TotalCharges` din setul de variabile, păstrând `tenure` și `MonthlyCharges` (conceptual distincte: vechime vs. cost lunar), pentru a reduce multicoliniaritatea și a obține coeficienți mai ușor de interpretat.

### Observație suplimentară

Mai multe variabile legate de "No internet service" (StreamingMovies, OnlineSecurity, DeviceProtection etc.) au coeficienți identici (-0.172521) — normal, fiindcă sunt perfect corelate (un client fără internet nu poate avea niciunul din aceste servicii, informația fiind redundantă).

## Multicoliniaritate — investigație extinsă

Eliminarea `TotalCharges` nu a rezolvat complet problema coeficientului negativ pentru `MonthlyCharges`. Motivul: `MonthlyCharges` e corelat și cu variabilele de servicii (StreamingTV, OnlineSecurity, MultipleLines, InternetService_Fiber optic, etc.) — clienții cu mai multe servicii plătesc mai mult, deci informația de cost e parțial "explicată" prin combinația acestor variabile.

**Concept relevant — paradoxul Simpson:** relația simplă (bivariată) dintre `MonthlyCharges` și `Churn`, observată în EDA, e pozitivă (corelație 0.19) — facturi mai mari, abandon mai mare. Dar relația condiționată (controlând pentru toate celelalte variabile simultan, inclusiv ce servicii are clientul) devine negativă în model. Nu e o eroare — modelul răspunde la o întrebare diferită: "între doi clienți cu exact aceleași servicii și contract, dar facturi diferite, care are risc mai mare?" — iar răspunsul izolat de context poate diferi de tendința generală brută.

**Decizie:** documentăm fenomenul ca o limitare cunoscută a interpretării coeficienților individuali în prezența multicoliniarității, dar continuăm cu acest model, fiindcă scopul principal e capacitatea predictivă (evaluată separat prin metrici de clasificare), nu interpretarea izolată a fiecărui coeficient.

## Evaluarea modelului — Logistic Regression (baseline)

### Confusion Matrix

|  | Predicție: No Churn | Predicție: Churn |
|---|---|---|
| **Real: No Churn** | 918 (TN) | 117 (FP) |
| **Real: Churn** | 170 (FN) | 204 (TP) |

### Metrici

| Metric | No Churn | Churn |
|---|---|---|
| Precision | 0.84 | 0.64 |
| Recall | 0.89 | 0.55 |
| F1-score | 0.86 | 0.59 |

**Accuracy generală: 80%**

### Interpretare

Modelul are accuracy de 80%, doar cu 6.5 puncte peste un model "naiv" care ar prezice mereu "No Churn" (baseline de 73.5%, dat fiind dezechilibrul de clase). 

Pentru clasa de interes (Churn):
- **Precision 0.64** — din clienții marcați ca risc de abandon, 64% abandonează efectiv
- **Recall 0.55** — modelul identifică doar 55% din clienții care abandonează efectiv, ratând 45% (170 din 374 clienți reali la risc, clasificați greșit ca "No Churn")

**Concluzie de business:** modelul are o capacitate predictivă utilă, dar limitată — ratează aproape jumătate dintre clienții care abandonează real, ceea ce ar limita eficiența unor acțiuni proactive de retenție bazate exclusiv pe acest model. Recall-ul scăzut e cauzat probabil de dezechilibrul de clase din dataset (26.5% Churn), care face ca modelul să fie predispus să prezică mai des clasa majoritară.

**Pași următori de explorat:** ajustarea threshold-ului de decizie, `class_weight='balanced'`.

## Class Weight Balanced — comparație trade-off

Am reantrenat modelul cu `class_weight='balanced'`, care penalizează mai mult erorile pe clasa minoritară (Churn) în timpul antrenării, compensând dezechilibrul natural al claselor (26.5% Churn vs 73.5% No Churn).

### Comparație directă (clasa Churn)

| Metric | Fără balanced | Cu balanced |
|---|---|---|
| Precision | 0.64 | 0.51 |
| Recall | 0.55 | 0.77 |
| F1-score | 0.59 | 0.61 |
| Accuracy generală | 80% | 74% |
| False Negative (clienți la risc ratați) | 170 | 85 |
| False Positive (alarme false) | 117 | 282 |

### Interpretare — trade-off precision vs recall

`class_weight='balanced'` crește semnificativ recall-ul (de la 55% la 77%), reducând la jumătate numărul de clienți reali la risc ratați de model (de la 170 la 85). În schimb, precision-ul scade (de la 64% la 51%), dublând aproape numărul de alarme false.

**Decizia depinde de costul de business al celor două tipuri de erori:**
- Dacă acțiunile de retenție (discount, contact proactiv) sunt ieftine, varianta `balanced` e preferabilă — prioritizează prinderea cât mai multor clienți la risc real, acceptând mai multe alarme false.
- Dacă acțiunile de retenție sunt costisitoare, varianta fără `balanced` e mai prudentă —