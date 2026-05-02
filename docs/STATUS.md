# Статус проекта DOTM Sniper

## Дата: 2026-05-01

## Что работает

- Сбор рынков через `pm-trader markets list --limit 200` ✅
- Фильтрация по цене < $0.04 ✅
- Топ-10 по объёму ✅
- OpenRouter API вызовы (статус 200) ✅
- Исправлена команда `pm-trader buy` — теперь `pm-trader buy <slug> <outcome> <amount>` ✅
- Обработка `content: null` — читает из `reasoning` если `content` пустой ✅
- Снижены пороги: edge > 0.01, confidence ≥ 2 (для тестирования) ✅

## Главная проблема

**MiniMax не возвращает `content` на русском промпте.**

При запросе с русским промптом (2138 символов, 10 рынков) API возвращает:
- `choices[0].message.content: null`
- `choices[0].message.reasoning: "The user is asking for a specific output..."` (500+ символов текста)
- То есть текст кладётся в `reasoning` вместо `content`

При английском промпте (коротком) — `content` содержит текст нормально.

**Гипотеза:** MiniMax на длинных промптах (>2000 токенов) кладёт текст в `reasoning` но не в `content`.

## Логи

```
DEBUG: status=200
DEBUG: response=... (JSON с 32582 символами, включая reasoning)

Message keys: ['role', 'content', 'refusal', 'reasoning', 'reasoning_details']
Content: None
Reasoning: 'The user is asking for a specific output: a JSON array of objects...'
```

## Что проверить

1. Попробовать английский промпт (короче) — извлечется ли JSON из `reasoning`?
2. Попробовать модель `anthropic/claude-3.5-haiku` для сравнения
3. Проверить, есть ли в `reasoning` настоящий JSON массив с анализом рынков
4. Рассмотреть вариант парсинга `reasoning` если `content` пустой

## Команды pm-trader

```bash
# Список рынков
pm-trader markets list --limit 200

# Купить (правильный синтаксис!)
pm-trader buy <slug> <outcome> <amount_usd>
# Пример: pm-trader buy will-eth-reach-5000-2026 Yes 10

# Портфель
pm-trader portfolio

# Помощь
pm-trader buy --help
```

## API ключ OpenRouter

```
REDACTED_OPENROUTER_KEY
```

Модель: `minimax/minimax-m2.7-20260318`
URL: `https://openrouter.ai/api/v1/chat/completions`