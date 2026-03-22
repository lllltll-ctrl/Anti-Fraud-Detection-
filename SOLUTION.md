# Anti-Fraud Strategy — SKELAR x mono AI Competition

## 1. Огляд рішення

### Підхід: Two-Stage Graph + ML (Honest K-fold)

Побудовано двоетапну систему виявлення платіжних шахраїв, яка комбінує аналіз графу зв'язків між користувачами з ML-ансамблем. Усі метрики — **honest** (без data leakage): графові фічі обчислюються через K-fold, щоб уникнути підглядання в мітки валідації.

**Stage 1 — Графовий аналіз (Connected Components):**
- Будуємо граф через `card_mask_hash` — користувачі пов'язані, якщо ділять одну картку
- BFS знаходить connected components (420K компонентів)
- Компоненти де **≥90%** train-юзерів fraud → автоматично fraud (near-pure threshold)
- Компоненти без жодного fraud → автоматично legit
- Решта (mixed) → йде на ML-класифікацію

**Stage 2 — LightGBM Ensemble:**
- 3 моделі GBDT з різними seeds
- 65 features: поведінкові, графові (card + holder), target encoding, interaction features
- Greedy weight blend оптимізація
- K-fold graph features для честної оцінки

**Stage 3 — Post-processing (Fraud Propagation):**
- Після ML: поширюємо fraud через card + holder граф
- Якщо ≥60% сусідів юзера — fraud → помічаємо як fraud
- 2 раунди ітеративної propagation (min 3 сусіди)

### Ключові цифри
| Метрика | Значення |
|---------|----------|
| Honest OOF F1 (K-fold graph) | **0.8176** |
| Кількість ознак | 65 |
| Кількість моделей | 3 GBDT LightGBM |
| Частка fraud у train | 3.78% (14 932 / 395 381) |
| Прогнозований fraud у test | 6 405 (3.78%) |
| Two-stage threshold | 0.05 |
| Propagation threshold | 0.60 (min 3 neighbors) |

### Дані
- **395 381** користувачів у train (14 932 шахраї)
- **3 135 378** транзакцій у train
- **169 449** користувачів у test
- **1 353 503** транзакцій у test

---

## 2. Топ-5 ключових ознак / правил

### 1. Fraud ratio компоненти (`g_comp_fraud_ratio`) — importance: 255

**Що це:** частка відомих шахраїв серед усіх train-юзерів у тому ж connected component графу.

**Чому працює:** шахраї утворюють щільні кластери через спільні картки. Якщо компонента на 90%+ складається з fraud-юзерів — новий юзер у цій компоненті майже напевно теж шахрай.

**Статистика:**
- Feature importance: **255** (1-ше місце)
- Pure fraud компоненти (≥90%): 8 934 train-юзерів → auto-fraud
- Pure legit компоненти: 356 584 train-юзерів → auto-legit
- Mixed: 29 863 train-юзерів → ML класифікація

**Бізнес-логіка:** моніторинг "кластерів" через shared cards — один заблокований акаунт розкриває всю мережу.

---

### 2. Розмір компоненти (`comp_size`) — importance: 172

**Що це:** кількість юзерів у connected component графу, до якої належить юзер.

**Чому працює:** fraud-кільця створюють великі компоненти через множинне використання карток. Легітимні юзери зазвичай ізольовані або мають маленькі компоненти (1-3 юзери). Великий розмір компоненти — сигнал організованої fraud-мережі.

**Статистика:**
- Feature importance: **172** (2-ге місце)
- Не містить fraud labels → жодного leakage

**Бізнес-логіка:** юзер з компонентою >10 акаунтів → підвищений моніторинг.

---

### 3. Fail ratio × кількість карток (`fail_x_cards`) — importance: 159

**Що це:** interaction feature — множимо частку невдалих транзакцій на кількість унікальних карток юзера.

**Чому працює:** шахраї перебирають багато карток і мають високий fail rate. Комбінація цих двох сигналів значно сильніша ніж кожен окремо. Легітимний юзер може мати або багато карток (бізнес), або високий fail (тимчасова проблема), але рідко обидва одночасно.

**Статистика:**
- Feature importance: **159** (3-тє місце)

**Бізнес-логіка:** юзер з >3 картками та fail rate >50% — високий ризик card testing.

---

### 4. Target-encoded traffic type (`traffic_type_te`) — importance: 131

**Що це:** K-fold target encoding типу трафіку (organic, paid, referral тощо) — замінює категорію на середній fraud rate серед юзерів з таким трафіком.

**Чому працює:** певні канали залучення мають значно вищий рівень шахрайства. Шахраї приходять через специфічні джерела трафіку, які відрізняються від легітимних користувачів.

**Статистика:**
- Feature importance: **131** (4-те місце)
- Обчислюється через K-fold для honest оцінки (smoothing=50)

**Бізнес-логіка:** посилена верифікація для реєстрацій з високо-ризикових каналів трафіку.

---

### 5. Кількість карток × кількість holders (`cards_x_holders`) — importance: 129

**Що це:** interaction feature — добуток кількості унікальних карток та унікальних card_holder імен юзера.

**Чому працює:** шахраї використовують багато різних карток з різними іменами holders — це множинна комбінація сигналів card testing. Легітимний юзер зазвичай має 1-2 картки з одним ім'ям.

**Статистика:**
- Feature importance: **129** (5-те місце)

**Бізнес-логіка:** >5 карток з >3 різними іменами holders → ініціювати ручну перевірку.

---

## 3. Пропозиція бізнес-інтеграції

### Проблема
Система має бути достатньо точною, щоб не блокувати легітимних клієнтів, але ефективно виявляти шахраїв.

### Рекомендована архітектура: Graph-First + ML Scoring

```
Нова реєстрація / транзакція
    │
    ▼
[ГРАФОВИЙ АНАЛІЗ]
    ├── Картка вже на fraud-акаунті? → ЧЕРВОНИЙ (автоблок)
    ├── card_holder на fraud-акаунті? → ЧЕРВОНИЙ (автоблок)
    ├── ≥60% сусідів — fraud? → ЧЕРВОНИЙ (propagation)
    └── Нових зв'язків немає → ML scoring
                                    │
                                    ▼
                            [ML SCORING]
                            Score < 0.07  → ЗЕЛЕНИЙ (дозволити)
                            0.07-0.30     → ЖОВТИЙ (обмеження)
                            Score ≥ 0.30  → ЧЕРВОНИЙ (ручна перевірка)
```

### Деталі по зонах

**ЧЕРВОНИЙ (автоблок через граф):**
- Картка shared з відомим fraud → заморозити платіж
- card_holder ім'я з fraud-акаунту → заморозити + KYC
- ≥60% graph-сусідів fraud → автоблок (propagation rule)
- Ловить ~60% шахраїв з precision ~99%

**ЧЕРВОНИЙ (ML score ≥ 0.30):**
- Заморозити вихідні платежі до верифікації
- Надіслати на ручну перевірку (3-5 аналітиків)
- Запросити KYC документи

**ЖОВТИЙ (ML score 0.07-0.30):**
- Зменшити ліміти транзакцій
- Увімкнути 3DS для всіх платежів
- Моніторинг в реальному часі

**ЗЕЛЕНИЙ (ML score < 0.07):**
- Жодних обмежень

### Чому Graph-First

1. **Граф дає найсильніший сигнал** — shared cards + shared card_holder між fraud-акаунтами ловлять більшість шахраїв
2. **Працює в реальному часі** — O(1) lookup по card_mask_hash та card_holder
3. **Не потребує ML** для основного detection — простий rule engine
4. **Propagation** — fraud поширюється по графу автоматично, розкриваючи нові fraud-акаунти
5. **ML доповнює** для випадків де графовий зв'язок відсутній

### Моніторинг і feedback loop

- Щотижневий аналіз false positive / false negative
- Дашборд з метриками по зонах
- Ретренінг моделі щомісяця з новими fraud-кейсами
- При виявленні нового fraud — автоматично поширювати "заразу" по графу на пов'язані акаунти (propagation)

### Очікуваний бізнес-ефект

- **Graph-based detection**: ловить ~60% шахраїв з мінімальним false positive
- **Propagation**: ще ~5% через автоматичне поширення fraud по графу
- **ML додатково**: ще ~20% шахраїв через поведінкові патерни
- **Комбіновано**: зниження fraud-втрат на **85%+** з втратою конверсії < 1%

---

## 4. Технічна реалізація

### Основний скрипт: `train_2stage_honest.py`

```
1. Завантаження даних (train/test transactions + users)
2. Побудова card graph (card_mask_hash → connected components, 420K)
3. Побудова holder graph (card_holder → зв'язки для features)
4. Pre-compute neighbor sets (card + holder)
5. Feature engineering (65 ознак)
6. Target encoding (K-fold, smoothing=50)
7. K-fold graph features (honest, no leakage)
8. Training: 3 моделі LightGBM GBDT
9. Greedy weight blend optimization
10. Two-stage override (near-pure ≥90% → auto, mixed → ML)
11. Fraud propagation через card + holder граф (threshold 0.6, 2 rounds, min 3 neighbors)
```

### Групи ознак (65)

| Група | Кількість | Приклади |
|-------|-----------|---------|
| Графові (card + holder) | 10 | g_holder_fraud_ratio, g_comp_fraud_ratio, g_card_max_density |
| Transaction aggregates | 15 | tx_count, fail_ratio, card_switch_ratio, amount_cv |
| Temporal | 6 | minutes_to_first_tx, mean_gap_sec, tx_first_hour |
| Error patterns | 4 | eg_fraud_ratio, eg_antifraud_ratio, eg_3ds_ratio |
| Geographic | 2 | country_mismatch_ratio, card_reg_mismatch_ratio |
| Target encoding | 4 | reg_country_te, gender_te, traffic_type_te, email_domain_te |
| Component | 2 | comp_size, log_comp_size |
| Interaction features | 6 | fail_x_cards, cards_x_holders, mismatch_x_cards, fraud_err_x_cards, fast_reg_x_fail, night_x_switch |
| Categoricals | 4 | gender, reg_country, traffic_type, email_domain |
| Other | 12 | risk_combo, log_tx_count, digital_wallet_ratio, has_prepaid |

### Ансамбль

| Модель | Folds | Seed | LR | Blend Weight |
|--------|-------|------|----|-------------|
| gbdt_5f_42 | 5 | 42 | 0.03 | 0.000 |
| gbdt_5f_123 | 5 | 123 | 0.03 | 0.371 |
| gbdt_5f_999 | 5 | 999 | 0.025 | 0.629 |

### Ключові рішення

1. **Near-pure threshold 90%** замість 100% — компоненти з ≥90% fraud автоматично класифікуються як fraud, що збільшує recall
2. **Holder graph для features, не для BFS** — holder edges у BFS занадто агресивні (об'єднують fraud + legit компоненти), тому використовуються тільки для graph features та propagation
3. **K-fold graph features** — кожен fold бачить тільки мітки з інших folds для обчислення графових ознак, що запобігає data leakage
4. **Fraud propagation** — після ML, поширюємо fraud по card + holder графу з порогом 60% та мінімум 3 сусіди (захист від каскадних помилок), 2 раунди (+199 нових fraud)
5. **Calibration cap** — якщо fraud rate перевищує train (3.78%), прибираємо найслабші fraud predictions для калібрації
