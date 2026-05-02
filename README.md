# DOTM Sniper

Бот-снайпер для Polymarket на базе MiniMax через OpenRouter.

## Структура проекта

```
dotm-sniper/
├── README.md          # Этот файл
├── src/
│   └── dotm_sniper.py   # Основной скрипт
└── docs/
    └── STATUS.md     # Статусы и логи
```

## Как работает

1. **Сбор рынков** — `pm-trader markets list --limit 200`, фильтр по цене < $0.04, топ-10 по объёму
2. **Анализ** — POST к OpenRouter (`minimax/minimax-m2.5:free`) с промптом на русском, просит JSON-массив с `estimated_probability`, `action`, `confidence`, `reasoning`
3. **Извлечение JSON** — поиск `[` ... `]`, при неудаче — regex для отдельных объектов
4. **Сделки** — если `action == "BUY"`, край > 0.01 и уверенность ≥ 2, вызов `pm-trader buy <slug> <outcome> <amount>`
5. **Цикл** — каждые 30 минут

## Запуск

```bash
cd dotm-sniper/src
python3 dotm_sniper.py
```

## Текущие проблемы

| Проблема | Статус |
|:---|:---|
| Формат вывода `pm-trader` — парсинг таблица→JSON | ✅ Решено |
| Команда `openclaw agent` — тайм-аут с длинными промптами | ✅ Решено |
| Пустой `content` в ответе MiniMax — модель кладёт текст в `reasoning` вместо `content` | ✅ Решено (теперь читает из `reasoning`) |
| **MiniMax не возвращает `content` на русском промпте** — приходит `content: null`, текст в `reasoning` | ❌ Открыто |
| **Команда `pm-trader buy`** — синтаксис `pm-trader buy SLUG OUTCOME AMOUNT_USD`, а не `--market-id` | ✅ Исправлено |

## Найденные нюансы

- `pm-trader buy` принимает `SLUG OUTCOME AMOUNT_USD`, например: `pm-trader buy will-eth-reach-5000-2026 Yes 10`
- MiniMax на русском промпте возвращает `content: null` но пишет текст в `reasoning`
- Перед покупкой нужно подтвердить, что модель реально выдаёт JSON в `content` или грамотно извлекать из `reasoning`

## Следующие шаги

1. Проверить что MiniMax выдаёт JSON-массив именно в `content` при английском промпте
2. Снизить порог уверенности до 2-3 (сейчас 5) для тестирования
3. Попробовать модель `anthropic/claude-3.5-sonnet` для сравнения качества
4. Добавить логирование в файл