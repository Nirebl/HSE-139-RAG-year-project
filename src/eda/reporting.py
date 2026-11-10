"""Russian findings derived from exported measurements, never expected values."""
import json

import pandas as pd

from .loaders import ROOT


def read(name):
    return json.loads((ROOT/'reports'/name).read_text(encoding='utf-8'))


def sections():
    r=read('results.json'); qa=r['qa']; protocol=read('embedding_protocol.json')
    features=pd.read_csv(ROOT/'reports/document_features.csv')
    structure=pd.read_csv(ROOT/'reports/structure_summary.csv').set_index('source_type')
    aug=pd.read_csv(ROOT/'reports/augmentations.csv')
    overlap=pd.read_csv(ROOT/'reports/vocabulary_overlap.csv',index_col=0)
    pairs=pd.read_csv(ROOT/'reports/semantic_pair_summary.csv')
    errors=read('download_errors.json'); exclusions=read('exclusions.json')
    med=features.groupby('source_type').tokens.median()
    reviews=pd.read_csv(ROOT/'reports/coverage_review.csv',keep_default_na=False)
    licenses='Kubernetes: CC-BY-4.0; GitLab: текст разрешения MIT в LICENSE (без заголовка MIT); NVIDIA: Apache-2.0 в card. Для отдельных postmortems свободная лицензия не установлена: только локальная исследовательская копия, без разрешения на перераспространение. Это фиксация опубликованных сведений, а не юридическое заключение. IBM README ведёт на PrimeQA/TechQA (800 тысяч TechNotes); полный upstream не загружался. Условия upstream не подменяются лицензией NVIDIA.'
    result=[
      ('Введение и исследовательские вопросы',
       'Исследуем пригодность четырёх типов источников для Incident Copilot. Гипотезы: структура и длина требуют разного chunking; общие технологии дают точки пересечения; точные идентификаторы важны для поиска; аугментации меняют представления. Это EDA, а не готовая RAG-система. Seed=42; объёмы и параметры в config/eda.json.'),
      ('Источники и ограничения выборки',
       f"После обработки: {r['counts']}; QA-пар: {qa['rows']}. {licenses}\n\nВыборка Kubernetes стратифицирована по разделу и квартилю размера; GitLab наполовину направлена на общую инфраструктуру; postmortems отбирались из каталога danluu по организациям. Это управляемая выборка, не доказанная статистическая репрезентативность. Исключено {len(exclusions)} записей; ошибок/неудачных проб загрузки: {len(errors)}. Allegro вернул короткую страницу перенаправления; полный отчёт не получен и не использован. Другие отчёты представляют несколько организаций."),
      ('Загрузка, предобработка и единая схема',
       'Оригинальные HTTP-тела сохраняются в data/raw с SHA-256, временем получения и URL. Отдельно сохраняются parsed, cleaned и processed. QA хранится отдельно со ссылками context_ids. ID зависят от типа и идентичности источника; для QA-контекстов также от содержания, чтобы не перезаписывать варианты одного filename. metadata.field_status различает observed/not_published/unknown; not_applicable и lost_in_parsing резервируются для подтверждённых случаев, искусственно не назначаются. Неизвестная версия остаётся null: номер в тексте не означает версию всего документа. Техническая пунктуация и регистр исходного содержания сохраняются.'),
      ('Аудит качества',
       f"Найдено {r['exact_pairs']} пар точных и {r['near_pairs']} суммарных пар-кандидатов дубликатов при cosine char 3–5 gram TF-IDF ≥0.95. Это кандидаты, не основание автоматически удалять документы. Символ U+FFFD встречается в {r['encoding_affected']} документах: повреждения уже присутствуют в данных, молча исправлять их нельзя. Пропуски показаны в абсолютных числах и долях; отсутствие даты публикации отличается от сбоя парсинга. HTML/Hugo-артефакты и кандидаты устаревания — эвристики для проверки. Длинный текст не считается ошибкой. Потенциальные противоречия по версиям не установлены автоматически: требуется экспертная сверка."),
      ('Technical QA: разметка и leakage',
       f"Фактические splits: {qa['splits']}; префиксы ID: {qa['id_prefixes']}. Разметка is_impossible: {qa['answerability']}; строго пустых ответов: {qa['empty_answers']}; с учётом placeholder (-/n/a/none): {qa['placeholder_answers']}, из них у answerable: {qa['answerable_placeholder_answers']}. Непустой дефис не считается размеченным содержательным ответом. Распределение числа контекстов: {qa['contexts_histogram']}. Префиксы TRAIN/DEV не превращаются в придуманные HF splits. Групп точных дубликатов вопросов: {len(qa['duplicate_question_groups'])}; межпрефиксных: {len(qa['cross_prefix_duplicate_questions'])}; контекстов, общих между префиксами: {qa['context_shared_across_prefixes']}. У answerable непустой содержательный ответ не найден как точная подстрока связанного контекста в {qa['context_not_answer_span']} случаях; это сигнал аудита, не доказательство ошибочной разметки. Near-duplicates вопросов сохранены в question_duplicate_pairs.csv.\n\nNVIDIA не публикует здесь полный исходный IBM corpus и официальный test. Вопросы IBM про WebSphere/Netcool/DataPower нельзя считать запросами к инфраструктуре GitLab. Контексты без размеченного положительного ответа не объявляются отрицательными. Для downstream QA есть answerability и ответы, для диагностики и cross-source similarity эталонных меток пока нет."),
      ('Техническая документация',
       f"Kubernetes: медиана {med['documentation']:.0f} WordPiece-токенов; среднее число заголовков {structure.loc['documentation','headings']:.1f}; средняя доля fenced code {structure.loc['documentation','code_fraction']:.1%}. Ссылки и Hugo-шаблоны сохраняются и измеряются. Семпл включает reference/concepts/tasks и вклад в документацию, поэтому часть страниц не посвящена диагностике. Перед расширением корпуса нужно отделять вклад/стили от эксплуатационных материалов, сохраняя исходные пути и граф ссылок."),
      ('Runbooks: действия и условия',
       f"GitLab: медиана {med['runbook']:.0f} токенов; среднее {structure.loc['runbook','commands']:.1f} вхождений команд из фиксированного словаря; условия обнаружены в {structure.loc['runbook','conditions']:.1%}, ожидаемые результаты — в {structure.loc['runbook','expected_result']:.1%}. Это эвристическое наличие маркеров, не оценка полноты инструкции. Пустые/короткие переадресации исключены и перечислены. Текст MIT LICENSE сохранён. Инструкции привязаны к GitLab: команды требуют проверки окружения и прав. Сохранять блок «условие → действие → проверка» при будущем chunking."),
      ('Исторические инциденты',
       f"Принято {r['counts']['incident']} полных страниц отчётов, медиана {med['incident']:.0f} токенов. Маркеры timeline встречаются в {structure.loc['incident','timeline']:.1%}, root cause в {structure.loc['incident','root_cause']:.1%}, corrective actions в {structure.loc['incident','corrective_actions']:.1%}. Эти признаки эвристические и могут срабатывать на обычное слово cause. Два GitLab-отчёта описывают один инцидент: число страниц не равно числу независимых событий. Кандидатные категории — DNS, база/репликация, deployment/configuration; это не gold-классы. Один AWS-ответ содержит корейский текст: сохранён в EDA, исключён из англоязычного embedding-анализа. Требуется лучшая дедупликация по событию и проверка прав каждого издателя."),
      ('Cross-source и NLP',
       f"Медианы WordPiece-токенов: {med.to_dict()}. Jaccard словарей GitLab/Kubernetes при равном бюджете до 10 000 токенов: {overlap.loc['runbook','documentation']:.3f}; IBM/Kubernetes: {overlap.loc['qa_context','documentation']:.3f}. Равный бюджет уменьшает влияние размеров, но не устраняет разницу жанров. Токенизация сохраняет CVE, пути, версии и параметры; technical_density — доля lexical tokens с цифрами, пунктуацией идентификатора или аббревиатурами, а не терминологический gold. Сохранены hapax, TTR и частотные n-граммы. Частотные служебные слова не удалялись из содержания; чисто лексическое пересечение не измеряет качество RAG."),
      ('Нейронное пространство',
       f"Использована {protocol['model']} revision {protocol['revision']}, CPU, mean pooling, L2, batch={protocol['batch_size']}, max_seq_length={protocol['max_length']}. {r['truncation_fraction']:.1%} документов превышают лимит с учётом special tokens. Вместо молчаливого усечения сформировано {r['chunks']} окон по {protocol['chunk_tokens']} токенов с overlap={protocol['overlap']}, с token/char offsets. Эмбеддинги {r['embedding_documents']} документов — нормированное среднее окон, что может размывать редкий симптом. Пары сравниваются на уровне документа; self-пары и точные дубликаты исключены, отдельные окна одного документа не сравниваются. По каждому сочетанию типов не более 500 пар (для малого incident-корпуса меньше).\n\nPCA даёт описание геометрии, не доказательство релевантности. Cosine не является вероятностью. QA-to-source maxima экспортированы отдельно; общий термин SSL или database не делает IBM-контекст инструкцией для GitLab. Модель англоязычная; для русского пользовательского ввода нужен отдельный multilingual-эксперимент."),
      ('Knowledge coverage: материалы для человека',
       f"Созданы 10 синтетических сценариев инфраструктурной диагностики и {r['coverage_rows']} кандидатов (до трёх на тип и запрос). Поиск выполнен по общему сохранённому корпусу английских инфраструктурных документов, без IBM и без включения ответов в индекс. Для каждого кандидата сохранены URL и наиболее близкое окно-доказательство. Проверено человеком: {r['human_reviewed']}/{r['coverage_rows']}. **Ожидает ручной проверки.** Предварительный статус unverified_candidate не означает релевантность. Не найденное среди кандидатов нельзя считать отсутствующим во всём корпусе.\n\nЗаполнить human_label: 0 — не помогает, 1 — частично, 2 — непосредственно; reviewer, reviewed_at и comment. Проверять полный документ, соответствие продукта/версии и полезность фрагмента. После этого можно подтвердить или опровергнуть дополнение источников; до проверки гипотеза остаётся открытой."),
      ('Аугментации и domain shift',
       '\n\n'.join(f"{method}: {len(g)} примеров; средний cosine к исходному {g.cosine.mean():.3f}; среднее изменение длины {g.length_delta.mean():.1f} символа; совпадение консервативного набора сущностей {g.entity_subset_preserved.mean():.1%}." for method,g in aug.groupby('method')) + '\n\nПерефразирования созданы агентом и явно синтетические; опечатки — перестановка соседних букв только в разрешённом обычном слове. Автоматический regex не исчерпывает сущности: несовпадения перечислены, смысл требует ручной проверки. Варианты не добавлены в исходные splits и не использованы для настройки. Высокий cosine не доказывает улучшение retrieval. Конкретный domain shift: вопрос TRAIN_Q016 просит решить «Too many open files» в DASH systemOut; Kubernetes использует нормативный API/CLI-стиль, а GitLab — окружение и команды оператора. Совпадение общей ошибки не подтверждает переносимость решения.'),
      ('Простой retrieval baseline',
       'Реально выполнены BM25 и dense candidate search по одному общему корпусу. Состав корпуса, query IDs и пустые qrels сохранены в retrieval_protocol.json, top-3 и их пересечения — retrieval_candidates.csv. Численные Recall/MRR/nDCG не считаются: проверенной разметки пока нет. После оценки человека разметка останется неполной (pool top-3), поэтому метрики можно интерпретировать лишь относительно judged pool. Для IBM нужен отдельный benchmark с полным корпусом/официальным split. Hard negatives не формировались: нерелевантность кандидатов не установлена.'),
      ('EDA Findings → RAG Architecture Decisions',
       f"| Наблюдение и доказательство | Риск | Решение | Следующий эксперимент |\n|---|---|---|---|\n| {r['truncation_fraction']:.1%} текстов длиннее окна MiniLM | Потеря свидетельств при truncation | Разбиение по структуре, offsets и parent ID | Сравнить fixed windows/heading-aware по human qrels |\n| U+FFFD в {r['encoding_affected']} документах | Точные идентификаторы могут быть повреждены | Флаг качества и исходный текст | Вручную проверить повреждённые technical spans |\n| {r['near_pairs']} пар кандидатов дубликатов; общие контексты TRAIN/DEV | Leakage и завышение разнообразия | Group split по документу и событию | Переоценить retrieval после группировки |\n| Версии и даты часто не опубликованы отдельными полями | Неподтверждённая применимость инструкции | Явная неизвестная версия, product/source фильтры | Разметить версии и проверить фильтрацию |\n| Runbooks содержат условия ({structure.loc['runbook','conditions']:.1%} по эвристике) | Отрыв команды от предпосылок | Сохранять последовательность диагностики | Human оценка полноты retrieved step bundle |\n| Есть неразрешимые IBM QA: {qa['answerability'].get('True',0)} | Ответ без доказательства | Отдельный abstention benchmark | Калибровать только на отдельной dev-разметке |\n| Человек ещё не подтвердил coverage | Ложная уверенность по cosine | Evidence review и qrels до claims | Проверить CSV, затем оценивать дополнение источников |"),
      ('Ограничения, критерии и следующие шаги',
       'Код, реальные данные, сравнения, нейронные эмбеддинги, графики и два augmentation-эксперимента выполнены. Проверка человеком knowledge coverage остаётся обязательной незавершённой частью. Семантическая проверка перефразирований также не заменяется regex/cosine. Статус исполнения ноутбука фиксируется отдельно в reports/notebook_execution.json. Полная приёмка ТЗ до human review не заявляется. Официальная рубрика курса в репозитории не найдена: соответствие внешнему DL-чекпойнту не подтверждено.\n\nКоманде: расширить корпус операционных Kubernetes docs, разметить 10 coverage-запросов, сгруппировать постмортемы по событиям, проверить версии и лицензии издателей, построить отдельный IBM evaluation split, затем сравнить hybrid retrieval и reranking. Не следует переносить совпадение терминов в утверждение о качестве диагностики.')
    ]
    example_path=ROOT/'reports/cross_source_examples.md'
    if example_path.exists():
        title,body=result[10]
        result[10]=(title,body+'\n\n'+example_path.read_text(encoding='utf-8').replace('# Смысловой разбор кандидатов агентом','### Предварительный разбор кандидатов агентом',1))
    semantic=pairs.set_index(['source_a','source_b'])['mean']
    title,body=result[9]
    result[9]=(title,body+f"\n\nСредний cosine: Kubernetes/Kubernetes {semantic.loc[('documentation','documentation')]:.3f}, Kubernetes/GitLab {semantic.loc[('documentation','runbook')]:.3f}, incidents/GitLab {semantic.loc[('incident','runbook')]:.3f}, IBM/GitLab {semantic.loc[('qa_context','runbook')]:.3f}. Это подтверждает геометрический domain shift в данном представлении, но не сравнивает качество RAG.")
    return result


def write_summary():
    text='# EDA Incident Copilot: фактические результаты\n\n'
    text+='\n\n'.join('## '+title+'\n\n'+body for title,body in sections())
    status_path=ROOT/'reports/notebook_execution.json'
    execution=json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
    execution_status='свежий kernel: PASS; ошибок '+str(execution.get('errors')) if execution.get('status')=='passed' else 'ещё не подтверждено; см. notebook_execution.json'
    criteria=[
      ('Реальные документированные источники','выполнено'),('Четыре категории','выполнено'),
      ('Пропуски, ошибки, дубликаты','выполнено'),('Длины, структура, NLP','выполнено'),
      ('Доступная разметка','выполнено'),('Сравнение источников','выполнено'),('Единая схема','выполнено'),
      ('Перефразирование и опечатки','измерения выполнены; human semantics review не проводился'),
      ('Семантическая совместимость','нейронные измерения выполнены; релевантность не подтверждена человеком'),
      ('Проверяемый coverage','материалы подготовлены; ожидает ручной проверки'),
      ('Выводы для RAG','выполнено с ограничениями'),('Реальные статистики/графики','выполнено'),
      ('Исполнение ноутбука',execution_status),('Воспроизводимость','локальный snapshot + pinned model; см. README и notebook_execution.json'),
      ('Добавление новых данных','единая схема и инструкция data/README.md')]
    text+='\n\n## Статус каждого критерия §18\n\n| Критерий | Статус |\n|---|---|\n'
    text+='\n'.join(f'| {a} | {b} |' for a,b in criteria)
    (ROOT/'reports/summary.md').write_text(text+'\n',encoding='utf-8')


if __name__=='__main__':
    write_summary()
