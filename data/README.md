# Данные EDA

`source_manifest.json` фиксирует URL, revision, время получения UTC, SHA-256, размер, условия использования и выбранные документы. `config/eda.json` содержит seed=42, лимиты, модель и пороги. Выборка не заявляется статистически репрезентативной.

Слои локального снимка:

- `raw/`: оригинальные HTTP-тела под SHA-256 URL + JSON-sidecar; не редактировать.
- `parsed/`: QA-оригиналы и извлечённые тексты, до исключения коротких страниц.
- `cleaned/`: консервативная нормализация переводов строк, без удаления технических символов.
- `processed/`: единая схема документов, отдельные QA со ссылками context_ids, окна с offsets, эмбеддинги и порядок ID.
- `cache/`: эмбеддинги, привязанные к текстам, preprocessing, модели, revision и параметрам; локальный кэш matplotlib.

Установка и команды приведены в README проекта. При существующем manifest команда загрузки восстанавливает именно выбранный снимок и проверяет checksum: нельзя незаметно заменить изменившийся постмортем новым. Если издатель изменил страницу, требуется прежний локальный raw snapshot или отдельный явно задокументированный новый снимок. Полная offline-воспроизводимость требует сохранить raw/parsed/processed и models локально. Повторный анализ сети не требует.

## Источники и права

- [NVIDIA TechQA-RAG-Eval](https://huggingface.co/datasets/nvidia/TechQA-RAG-Eval): Apache-2.0 в card; единственный опубликованный train.json, 910 строк. Сохраняются TRAIN/DEV префиксы. Это производный benchmark IBM, не GitLab/Kubernetes QA.
- [IBM TechQA](https://github.com/IBM/techqa): README и revision изучены; README ссылается на PrimeQA/TechQA, включая большой корпус TechNotes. Полный upstream не загружался; LICENSE по ожидаемому пути не найден (ошибка сохранена). Это ограничение проверки upstream-прав.
- [Kubernetes website](https://github.com/kubernetes/website): документация CC-BY-4.0, LICENSE сохранён. Выбор до 100 английских Markdown по section × квартиль размера, seed42, round-robin. Ссылки, frontmatter и Hugo shortcodes сохраняются. Некоторые короткие файлы требуют Hugo include/render и исключены из полнотекстового анализа.
- [GitLab Runbooks](https://gitlab.com/gitlab-com/runbooks): pinned master revision; LICENSE содержит стандартное MIT-разрешение без строки «MIT License». До 60 Markdown: половина по путям postgres/redis/kube/patroni/dns, остальное seeded random. Исходный manifest консервативно пометил условия как local-only, потому что первоначальное распознавание искало заголовок; фактический текст LICENSE проверен и правило исправлено. Переадресации на внешние инструкции не заменяются выдуманным содержанием.
- [danluu/post-mortems](https://github.com/danluu/post-mortems): каталог ссылок, не корпус полных текстов. Получены страницы исходных издателей GitLab, GitHub, Cloudflare, Slack, AWS; Allegro вернул короткие redirect-страницы, исключённые из корпуса. Свободная лицензия полных отчётов не установлена; тексты и фрагменты используются только локально, разрешение на распространение не утверждается. Один AWS-ответ на корейском: сохранён и отмечен ko эвристикой Hangul, исключён из английского embedding-анализа.
- [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): модель и tokenizer зафиксированы по commit в manifest; Apache-2.0, английский, mean pooling, max256. Модель скачивается отдельно в models/minilm.

Raw/derived тексты, веса, notebook с фрагментами и review CSV добавлены в .gitignore до проверки прав. Ничего не опубликовано. Это намеренно локальные сдаваемые артефакты; перед передачей третьим лицам нужно проверить права или пересобрать notebook без защищённых фрагментов. Числовые отчёты и исходный код отделены от текстового корпуса.

## Добавление корпуса

1. Реализовать ограниченное получение и сохранить raw через `Snapshot.get`, с revision/условиями; добавить правило выбора в manifest.
2. Извлечь content, сохранив команды и разметку. Использовать `schema.document` с одним из четырёх source_type; специфические поля хранить в metadata.
3. Отсутствующие значения не выдумывать: указать `metadata.field_status` (not_published/unknown/not_applicable/lost_in_parsing по установленной причине).
4. Добавить строки в локальный `data/parsed/source_records.json` со структурой как у существующих записей: source_type, source_name, url, path (для репозитория), revision, terms, product, body. Зафиксировать источник и выбор отдельно в manifest.
5. Выполнить preprocessing, pipeline и notebook, проверить schema/QA ссылки и причины исключений. Базовая статистика и графики группируются по source_type автоматически. Для нового языка пересмотреть применимость модели до включения его в semantic sample.

Автоматический повторный run сохраняет человеческие оценки CSV только при совпадении query_id, document_id и evidence. Не считать cosine или preliminary_label оценкой человека. Не изменять исходный test и не настраивать retrieval по производным тестовым формулировкам.
