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