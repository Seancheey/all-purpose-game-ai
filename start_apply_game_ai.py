if __name__ == '__main__':
    from config import config

    game_ai = config.provide_ai_applier()
    game_ai.start_apply_keyboard_events()
