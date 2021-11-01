All Purpose Game AI
---

This is a generic game AI framework that allows you to record screen and keyboard events in the game and save them as
dataset. The dataset then could be used to train AIs to be applied on the game.

## Project Setup

### Python Setup

This project is written in Python3.9 . Install Python3.9, and install all dependencies in `requirements.txt` by:

```bash
python -m pip install -r requirements.txt
```

### Writing Config

You should modify the `config.py` file in the project root directory for the game you are targeting on, then
set `config = your_custom_config`.

## Data Collection

```bash
python start_record.py
```

## Visualize Data Collected

```bash
python start_visualize_data.py
```

## Model Training

```bash
python start_train.py
```

## Start Game AI Bot

```bash
python start_apply_game_ai.py
```