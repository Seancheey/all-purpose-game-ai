All Purpose Game AI
---

This is a generic game AI framework that allows you to record screen and keyboard events in the game and save them as
dataset. The dataset then could be used to train AIs to be applied on the game.

## Writing Config

You should modify the `config.py` file in the project root directory for the game you are targeting on, then
set `config = your_custom_config`.

## Data Collection

run

```bash
python start_record.py
```

## Model Training

run

```bash
python start_train.py
```

## Start Game AI Bot

run

```bash
python start_apply_game_ai.py
```